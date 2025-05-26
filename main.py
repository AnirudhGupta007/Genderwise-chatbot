import asyncio
import json
import logging
import os
import uuid
import time
from datetime import datetime, timedelta, date # Added date
from typing import Optional, Dict, Any, List
import aiofiles

# --- Import Request directly from starlette ---
from starlette.requests import Request
from starlette.responses import RedirectResponse, HTMLResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
import httpx
from starlette.middleware.sessions import SessionMiddleware

# FastAPI and related imports
from fastapi import (
    FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect,
    UploadFile, File, Form, APIRouter, status, Query, Request as FastAPIRequest, # Keep FastAPIRequest alias
    Response # Import Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse
# Security imports
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Pydantic imports
from pydantic import BaseModel, Field, EmailStr

# SQLAlchemy imports
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, ForeignKey, Boolean, inspect,
    func, text # Import func and text
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base

# Environment and other utils
from dotenv import load_dotenv

# --- Import application modules ---
try:
    # from db_manager import db_manager # Import deferred
    from auth_utils import (
        verify_password, get_password_hash, create_access_token, TokenData,
        ALGORITHM, SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES
    )
    from jose import JWTError, jwt
    from utils import (
        ModelManager, RAGManager, FallbackHandler, MemoryMonitor,
        create_error_response, generate_uuid, MAX_HISTORY_TURNS,
        MODEL_IDLE_TIMEOUT, PRELOAD_MODEL
    )
    # Removed translation imports
except ImportError as e:
    print(f"FATAL ERROR: Failed to import necessary modules. Check file names and paths.")
    print(f"Import Error: {e}")
    exit(1)


# --- Load Environment Variables ---
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Configuration & Setup ---
DATABASE_URL = "postgresql://chatbot_user:itm%402016@localhost:5432/chatbot_db"
if not DATABASE_URL:
    logger.critical("DATABASE_URL environment variable not set!")
    exit(1)

Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Background Task Management ---
background_tasks = set()
idle_check_task = None

# ===================================
# --- Database Models ---
# ===================================
class User(Base):
    __tablename__ = "users"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    uploaded_docs = relationship("UploadedDocument", back_populates="uploader")
    curated_responses_added = relationship("CuratedResponse", back_populates="added_by_user")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False, default="New Conversation")
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True) # Used for soft delete
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

class Message(Base):
    __tablename__ = "messages"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    conversation_id = Column(PG_UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    original_language = Column(String, nullable=True, default="en") # Keep column, default to 'en'
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    message_metadata = Column(JSONB, nullable=True) # Store sources, etc.
    conversation = relationship("Conversation", back_populates="messages")

class UploadedDocument(Base):
    __tablename__ = "uploaded_documents"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    uploader = relationship("User", back_populates="uploaded_docs")

class CuratedResponse(Base):
    __tablename__ = "curated_responses"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    query_pattern = Column(Text, unique=True, nullable=False, index=True)
    response = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    added_by_user = relationship("User", back_populates="curated_responses_added")

# ===================================
# --- Pydantic Models for API ---
# ===================================
class Token(BaseModel):
    access_token: str
    token_type: str

class UserBase(BaseModel):
    email: EmailStr
    name: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserResponse(UserBase):
    id: uuid.UUID
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class MessageCreate(BaseModel): # No longer used directly by user endpoints
    content: str = Field(..., min_length=1)
    # language: str = "en" # Field removed/ignored

class ConversationCreate(BaseModel):
    title: str = Field(default="New Conversation", max_length=100)

class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    message_metadata: Optional[Dict[str, Any]] = None
    original_language: Optional[str] = None # Keep field for potential future use

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    messages: Optional[List[MessageResponse]] = []

    class Config:
        from_attributes = True

class UploadedDocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    uploaded_at: datetime

    class Config:
        from_attributes = True

class CuratedResponseResponse(BaseModel):
    id: uuid.UUID
    query_pattern: str
    response: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ===================================
# --- Dependency Functions ---
# ===================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def format_chat_history(messages: List[Message]) -> str:
    if not messages:
        return "No previous conversation history."
    history_str = ""
    # Only include last N turns
    start_index = max(0, len(messages) - MAX_HISTORY_TURNS * 2)
    for msg in messages[start_index:]:
        role = msg.role.upper()
        if role in ["USER", "ASSISTANT"]: # Ensure only user/assistant roles are included
            history_str += f"{role}: {msg.content}\n"
    return history_str.strip()

# --- Authentication Dependencies (Main App JWT) ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"},)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            logger.warning("Token missing 'sub' (email) claim.")
            raise credentials_exception
    except JWTError as e:
        logger.warning(f"JWT Error during token decode: {e}")
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        logger.warning(f"User specified in token not found: {email}")
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        logger.warning(f"Authentication attempt by inactive user: {current_user.email}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

# --- Admin Session Authentication Dependency ---
async def verify_admin_session(request: Request):
    """
    Checks if a valid admin session exists. If not, raises HTTPException
    which the exception handler (or default behavior) will use to redirect.
    """
    if not request.session.get("is_admin_logged_in"):
        logger.debug("Admin session verification failed. Redirecting to admin login.")
        login_url = request.url_for('admin_login_page')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # Use 401 to indicate auth needed
            detail="Not authenticated as admin.",
            headers={"Location": str(login_url)} # Suggest location for redirect
        )
    logger.debug(f"Admin session verified for user: {request.session.get('admin_user_email')}")

# ===================================
# --- FastAPI Application Setup ---
# ===================================
app = FastAPI( title="GenderWise RAG Chatbot API - V2.3 (No Translation, Analytics)", version="2.3.0", description="API for the GenderWise chatbot with JWT user auth and Session admin auth. Translation removed, basic analytics added." )
templates = Jinja2Templates(directory="Templates") # Setup templating

# OAuth Configuration
oauth = OAuth()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )
else:
    logger.warning("Google OAuth Client ID or Secret not configured. Google Login disabled.")

# Session Middleware
if not SECRET_KEY:
    logger.critical("FATAL: SECRET_KEY environment variable not set!")
    exit(1)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY) # Session middleware is crucial

# CORS Configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(',')
allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]
if not allowed_origins:
    allowed_origins = ["*"] # Default to allow all if not specified
logger.info(f"Configuring CORS for origins: {allowed_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods including DELETE
    allow_headers=["*"],
)

# --- Global Variables / Managers (Instantiate after models) ---
from db_manager import DBManager
db_manager = DBManager()
model_manager = ModelManager()
rag_manager = RAGManager(retriever=db_manager.get_retriever())

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, conversation_id: str):
        await websocket.accept() # Accept connection first
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = []
        if websocket not in self.active_connections[conversation_id]:
            self.active_connections[conversation_id].append(websocket)
            logger.info(f"WebSocket connection tracked for conversation {conversation_id}")
        else:
            logger.warning(f"Attempted to track already tracked websocket for conversation {conversation_id}")

    def disconnect(self, websocket: WebSocket, conversation_id: str):
        if conversation_id in self.active_connections:
            if websocket in self.active_connections[conversation_id]:
                self.active_connections[conversation_id].remove(websocket)
                logger.debug(f"Untracked specific websocket for conversation {conversation_id}")
            if not self.active_connections[conversation_id]: # Remove conversation key if list is empty
                del self.active_connections[conversation_id]
                logger.debug(f"Removed conversation {conversation_id} from active WS connections.")
        logger.info(f"WebSocket untracking processing complete for conversation {conversation_id}")

ws_manager = ConnectionManager()

# --- Supported Languages Removed ---

# --- Event Handlers (Startup/Shutdown) ---
async def periodic_idle_check():
    global model_manager
    while True:
        await asyncio.sleep(60) # Check every minute
        logger.debug("Running periodic idle check...")
        await model_manager.check_idle()

@app.on_event("startup")
async def startup_event():
    global idle_check_task
    logger.info("Application startup...")
    logger.info("Creating database tables if they don't exist...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables checked/created successfully.")
    except Exception as e:
        logger.critical(f"Failed to create database tables: {e}", exc_info=True)
        raise RuntimeError(f"Database initialization failed: {e}") from e

    if MODEL_IDLE_TIMEOUT > 0:
        logger.info(f"Starting background task for model idle check (timeout: {MODEL_IDLE_TIMEOUT}s)")
        idle_check_task = asyncio.create_task(periodic_idle_check())
        background_tasks.add(idle_check_task)
        # Ensure task is cleaned up properly on shutdown
        idle_check_task.add_done_callback(background_tasks.discard)
        if PRELOAD_MODEL:
            logger.info("Preloading model...")
            # Run preloading in background to not block startup fully
            asyncio.create_task(model_manager.get_llm_instance())
    else:
        logger.info("Model idle check disabled (MODEL_IDLE_TIMEOUT <= 0).")

    # Removed translation model loading
    logger.info("RAG Chatbot API started successfully.")
    mem_usage = MemoryMonitor.get_memory_usage()
    logger.info(f"Initial Memory Usage: {mem_usage}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    if idle_check_task and not idle_check_task.done():
        idle_check_task.cancel()
        try:
            await idle_check_task
        except asyncio.CancelledError:
            logger.info("Model idle check task cancelled.")
        except Exception as e:
            logger.error(f"Error during idle check task shutdown: {e}", exc_info=True)
    if model_manager.is_model_loaded():
        logger.info("Unloading model during shutdown...")
        await model_manager.unload_model()
    logger.info("RAG Chatbot API shutdown complete.")

# --- Exception Handler for Admin Redirect ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Specifically handle the 401 from verify_admin_session for redirection
    if exc.status_code == 401 and exc.detail == "Not authenticated as admin." and "Location" in exc.headers:
        return RedirectResponse(url=exc.headers["Location"], status_code=status.HTTP_303_SEE_OTHER)
    # Default handling for other HTTPExceptions (e.g., 404, 403, 500)
    # Ensure we return JSON for API errors
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}, # Standard FastAPI error format
            headers=exc.headers,
        )
    else: # For non-API routes (like admin HTML pages if an error happens there)
        # Could render an error template or just return plain text
        return HTMLResponse(
            content=f"<html><body><h1>Error {exc.status_code}</h1><p>{exc.detail}</p></body></html>",
            status_code=exc.status_code,
            headers=exc.headers
        )


# ===================================
# --- API Routers ---
# ===================================

# --- Authentication Router (Main App JWT Auth) ---
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@auth_router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    hashed_password = get_password_hash(user_data.password)
    db_user = User(email=user_data.email, name=user_data.name, hashed_password=hashed_password, is_admin=False)
    db.add(db_user)
    try:
        db.commit()
        db.refresh(db_user)
        logger.info(f"Registered new user: {user_data.email}")
        return db_user
    except Exception as e:
        db.rollback()
        logger.error(f"Error registering user {user_data.email}: {e}", exc_info=True)
        # Check for unique constraint violation specifically
        if "unique constraint" in str(e).lower() or "duplicate key value" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not register user.")

@auth_router.get('/login/google')
async def login_via_google(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google Login not configured.")
    redirect_uri = request.url_for('auth_google_callback')
    logger.debug(f"Redirecting to Google. Callback URI: {redirect_uri}")
    return await oauth.google.authorize_redirect(request, str(redirect_uri))

@auth_router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    logger.info(f"Generated token for user: {user.email}")
    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.post("/logout", status_code=status.HTTP_200_OK)
async def logout_user(current_user: User = Depends(get_current_user)):
    # This is mostly for acknowledgement, frontend handles token removal
    logger.info(f"Logout request acknowledged for user: {current_user.email}")
    # Future: Could add token to a blacklist here if using refresh tokens
    return {"message": "Logout acknowledged"}

@auth_router.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Gets the details of the currently authenticated user."""
    return current_user

app.include_router(auth_router)

# Google Callback Route (Main App JWT Auth)
@app.get('/api/auth/callback/google', name='auth_google_callback', include_in_schema=False)
async def auth_google_callback(request: Request, db: Session = Depends(get_db)):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google Login not configured.")
    try:
        token = await oauth.google.authorize_access_token(request)
        logger.info("Received token from Google.")
    except OAuthError as error:
        logger.error(f"Google OAuth Error: {error.error}")
        return HTMLResponse(f'<h1>Authentication Error</h1><p>{error.error}</p><p><a href="/">Go Home</a></p>', status_code=400)
    except Exception as e:
        logger.error(f"Error during Google token authorization: {e}", exc_info=True)
        return HTMLResponse(f'<h1>Internal Server Error</h1><p>Could not process Google login.</p><p><a href="/">Go Home</a></p>', status_code=500)

    user_info = token.get('userinfo')
    if not user_info:
        logger.error("Could not fetch user info from Google token.")
        return HTMLResponse(f'<h1>Authentication Error</h1><p>Could not retrieve user information from Google.</p><p><a href="/">Go Home</a></p>', status_code=400)

    google_email = user_info.get('email')
    google_name = user_info.get('name')

    if not google_email:
        logger.error("Email not found in Google user info.")
        return HTMLResponse(f'<h1>Authentication Error</h1><p>Email permission not granted or unavailable.</p><p><a href="/">Go Home</a></p>', status_code=400)

    logger.info(f"Google Login successful for user: {google_email} (Name: {google_name})")

    user = db.query(User).filter(User.email == google_email).first()

    if user:
        # Existing user
        logger.info(f"Existing user found for {google_email}")
        if not user.is_active:
            logger.warning(f"Google login attempt by inactive user: {google_email}")
            return HTMLResponse(f'<h1>Login Error</h1><p>Your account is inactive.</p><p><a href="/">Go Home</a></p>', status_code=400)
    else:
        # Create new user
        logger.info(f"Creating new user for {google_email} from Google login")
        # Generate a secure random password (user won't use it directly)
        hashed_password = get_password_hash(f"google_user_{uuid.uuid4()}")
        user = User(email=google_email, name=google_name, hashed_password=hashed_password, is_active=True, is_admin=False)
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
            logger.info(f"New user {google_email} created successfully.")
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create new user for {google_email}: {e}", exc_info=True)
            return HTMLResponse(f'<h1>Internal Server Error</h1><p>Could not create user account.</p><p><a href="/">Go Home</a></p>', status_code=500)

    # Generate JWT token for the user
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    app_access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)

    # Redirect user back to frontend with token (using JS to store in localStorage)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Logging In...</title></head>
    <body>
        <p>Login successful! Redirecting...</p>
        <script>
            const AUTH_TOKEN_KEY = 'genderwise_auth_token'; // Ensure this matches frontend key
            try {{
                localStorage.setItem(AUTH_TOKEN_KEY, "{app_access_token}");
                // Redirect to the main page (or dashboard)
                window.location.href = "/";
            }} catch (e) {{
                console.error("Failed to set token:", e);
                document.body.innerHTML = "<p>Login failed: Could not store session. Please enable cookies/localStorage and try again.</p><p><a href='/'>Go Home</a></p>";
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- Secured Conversation Router (Main App JWT Auth) ---
conv_router = APIRouter(prefix="/api/conversations", tags=["Conversations (Protected)"], dependencies=[Depends(get_current_active_user)])

@conv_router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(conversation_data: ConversationCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Creates a new conversation for the logged-in user."""
    try:
        new_conversation = Conversation(title=conversation_data.title, user_id=current_user.id)
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        logger.info(f"Created conversation {new_conversation.id} for user {current_user.email}")
        # Return response without messages initially
        conv_resp = ConversationResponse.from_orm(new_conversation)
        conv_resp.messages = []
        return conv_resp
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=create_error_response("Failed to create conversation")["detail"])

@conv_router.get("", response_model=List[ConversationResponse])
async def list_conversations(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Lists active conversations for the logged-in user."""
    try:
        # Get only active conversations, ordered by most recently updated
        conversations = db.query(Conversation).filter(
            Conversation.user_id == current_user.id,
            Conversation.is_active == True
        ).order_by(Conversation.updated_at.desc()).limit(100).all() # Limit results

        # Return list without messages for performance
        return [ConversationResponse(
                    id=c.id, title=c.title, created_at=c.created_at,
                    updated_at=c.updated_at, is_active=c.is_active, messages=None
                ) for c in conversations]
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=create_error_response("Failed to list conversations")["detail"])

@conv_router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Gets a specific conversation and its messages for the logged-in user."""
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
            # Allow fetching inactive conversations to view history? Let's allow it for now.
            # Conversation.is_active == True
        ).first()

        if not conversation:
            # Check if the conversation exists at all to differentiate 404 vs 403
            exists_check = db.query(Conversation.id).filter(Conversation.id == conversation_id).first()
            if exists_check:
                logger.warning(f"User {current_user.email} tried to access conversation {conversation_id} belonging to another user.")
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=create_error_response("Not authorized to access this conversation", "ACCESS_FORBIDDEN", 403)["detail"])
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=create_error_response("Conversation not found", "CONVERSATION_NOT_FOUND", 404)["detail"])

        # Fetch associated messages, limit the history fetched
        messages = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(Message.created_at).limit(200).all()

        conv_response = ConversationResponse.from_orm(conversation)
        conv_response.messages = [MessageResponse.from_orm(msg) for msg in messages]
        return conv_response
    except HTTPException as he:
        raise he # Re-raise specific HTTP exceptions
    except Exception as e:
        logger.error(f"Error retrieving conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=create_error_response("Failed to retrieve conversation")["detail"])

@conv_router.delete("/{conversation_id}", status_code=status.HTTP_200_OK)
async def delete_conversation(conversation_id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Soft deletes (marks as inactive) a conversation for the logged-in user."""
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()

        if not conversation:
            # Allow delete request for non-existent ID to be idempotent? Or return 404? Let's return 404.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=create_error_response("Conversation not found", "CONVERSATION_NOT_FOUND", 404)["detail"])

        if not conversation.is_active:
            # Already deleted, return success
             return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Conversation already deleted"})

        conversation.is_active = False
        conversation.updated_at = datetime.utcnow() # Explicitly update timestamp
        db.commit()
        logger.info(f"Soft deleted conversation {conversation_id} by user {current_user.email}")
        return {"message": "Conversation deleted successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=create_error_response("Failed to delete conversation")["detail"])

@conv_router.post("/{conversation_id}/messages", include_in_schema=False)
async def create_message():
    # Deprecated - use WebSocket
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Non-streaming endpoint deprecated. Use WebSocket.")

app.include_router(conv_router)

# --- Helper: Get User from Token for WebSocket (Main App JWT Auth) ---
async def get_user_from_token_ws(token: Optional[str] = Query(None), db: Session = Depends(get_db)) -> Optional[User]:
    if token is None:
        logger.debug("WS: Token not provided in query.")
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            logger.warning("WS: Token missing 'sub' claim.")
            return None
        # Use DB session passed as dependency
        user = db.query(User).filter(User.email == email).first()
        if user and user.is_active:
            return user
        elif user:
            logger.warning(f"WS: Auth attempt by inactive user: {email}")
            return None
        else:
            logger.warning(f"WS: User from token not found: {email}")
            return None
    except JWTError as e:
        logger.warning(f"WS: JWT Error: {e}")
        return None
    except Exception as e:
        logger.error(f"WS: Error validating token: {e}", exc_info=True)
        return None

# --- Secured WebSocket Endpoint (Main App JWT Auth) ---
# Use a unique path to avoid clashes if needed
@app.websocket("/ws/chat/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str, token: Optional[str] = Query(None)):
    # Create a unique DB session for this WebSocket connection's lifespan
    db = SessionLocal()
    try:
        current_user = await get_user_from_token_ws(token=token, db=db)
        if not current_user:
            await websocket.accept() # Accept briefly to send close code
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            logger.warning(f"WS: Unauthenticated connection attempt closed.")
            return

        try:
            conv_uuid = uuid.UUID(conversation_id)
        except ValueError:
            await websocket.accept()
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            logger.warning(f"WS: Invalid conversation ID format {conversation_id}. Closed.")
            return

        # Verify user owns the conversation and it exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conv_uuid,
            Conversation.user_id == current_user.id
            # Allow connecting to inactive conversations? Maybe not useful.
            # Conversation.is_active == True
        ).first()

        if not conversation:
            exists_check = db.query(Conversation.id).filter(Conversation.id == conv_uuid).scalar()
            close_code = status.WS_1008_POLICY_VIOLATION if exists_check else status.WS_1011_INTERNAL_ERROR
            await websocket.accept()
            await websocket.close(code=close_code)
            logger.warning(f"WS: Unauthorized/Not Found conv {conversation_id} for user {current_user.email}. Closed with code {close_code}.")
            return
        
        if not conversation.is_active:
             await websocket.accept()
             await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="Conversation is deleted.")
             logger.warning(f"WS: Attempt to connect to inactive conversation {conversation_id}. Closed.")
             return


        await ws_manager.connect(websocket, conversation_id) # Accepts inside connect now
        logger.info(f"WS: Connection active for conversation {conversation_id}, user {current_user.email}")

        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                content = message_data.get("content", "")
                # Language is removed - assume English
                language = "en" # Default implicitly
            except json.JSONDecodeError:
                logger.warning(f"WS: Invalid JSON received for conv {conversation_id}. Data: {data}")
                # Send error back via websocket if possible
                try: await websocket.send_json({"type": "error", "message": "Invalid message format."})
                except Exception: pass # Ignore if send fails
                continue
            if not content:
                continue

            start_time = time.time()
            logger.info(f"WS: Received message for conversation {conversation_id}: '{content[:50]}...'")

            # Save user message
            user_message = Message(
                conversation_id=conv_uuid,
                role="user",
                content=content,
                original_language=language # Store 'en'
            )
            try:
                db.add(user_message)
                db.commit()
                db.refresh(user_message)
                await websocket.send_json({"type": "message_received", "id": str(user_message.id)})
            except Exception as db_err:
                db.rollback()
                logger.error(f"WS: Error saving user message for conv {conversation_id}: {db_err}", exc_info=True)
                try: await websocket.send_json({"type": "error", "message": "Failed to save your message"})
                except Exception: pass
                continue

            # === Check for Curated Response ===
            curated_resp = db_manager.get_active_curated_response(content, db)
            if curated_resp:
                logger.info(f"WS: Found active curated response for query in conv {conversation_id}")
                assistant_text = curated_resp.response
                assistant_message_id = generate_uuid()
                # No translation needed
                assistant_message = Message(
                    id=assistant_message_id, conversation_id=conv_uuid, role="assistant",
                    content=assistant_text, original_language=language,
                    message_metadata={"source": "curated_response", "curated_id": str(curated_resp.id)}
                )
                db.add(assistant_message)
                # Update conversation timestamp
                conversation.updated_at = datetime.utcnow() # Update in-memory object timestamp
                db.commit() # Commit both message and conversation update
                await websocket.send_json({
                    "type": "complete", "message_id": str(assistant_message_id),
                    "content": assistant_text, "metadata": assistant_message.message_metadata
                })
                logger.info(f"WS: Sent curated response for conversation {conversation_id}")
                continue # Skip RAG/LLM

            # === Process message via RAG/LLM ===
            full_response_text = ""
            assistant_message_id = generate_uuid()
            final_metadata = None
            sources = []
            is_first_chunk = True
            stream_successful = False
            try:
                # Use content directly (no translation)
                content_en = content

                # Get recent history (limit turns)
                history_limit = MAX_HISTORY_TURNS * 2
                recent_messages = db.query(Message).filter(
                    Message.conversation_id == conv_uuid
                ).order_by(Message.created_at.desc()).limit(history_limit).all()
                recent_messages.reverse() # Put in chronological order for context
                formatted_history = format_chat_history(recent_messages)

                # Stream response using RAG Manager
                stream_iterator, sources = await rag_manager.stream_query_with_history(
                    question=content_en, chat_history=formatted_history
                )

                # Process the stream
                async for chunk_en in stream_iterator:
                    if chunk_en:
                        stream_successful = True
                        token_to_send = chunk_en # No translation
                        full_response_text += token_to_send
                        await websocket.send_json({"type": "stream", "token": token_to_send, "message_id": str(assistant_message_id), "is_first": is_first_chunk})
                        is_first_chunk = False

                logger.info(f"WS: Stream finished for conversation {conversation_id}")

                # Handle empty stream or fallback
                if not stream_successful:
                    logger.warning(f"WS: Stream yielded no text for conv {conversation_id}.")
                    full_response_text = FallbackHandler.get_fallback_response() # No language needed
                    await FallbackHandler.log_fallback(content, ValueError("Stream yielded no text"), str(conv_uuid))
                    await websocket.send_json({"type": "complete", "message_id": str(assistant_message_id), "content": full_response_text, "metadata": None})
                    # Still save the fallback message
                    assistant_message = Message(
                        id=assistant_message_id, conversation_id=conv_uuid, role="assistant",
                        content=full_response_text, original_language=language, message_metadata=None
                    )
                else:
                     # Prepare metadata if sources exist
                     if sources:
                         final_metadata = {"sources": [
                             {
                                 "title": doc.metadata.get("title", doc.metadata.get("original_source", "Unknown Source")),
                                 "source": doc.metadata.get("original_source", "Unknown"), # Prefer original filename
                                 "content_hash": doc.metadata.get("content_hash", None),
                                 "page_content_preview": doc.page_content[:150] + "..." # Use page content preview
                             } for doc in sources
                         ]}
                     else:
                         logger.info(f"WS: No RAG sources found/used for conv {conversation_id}.")

                     assistant_message = Message(
                         id=assistant_message_id, conversation_id=conv_uuid, role="assistant",
                         content=full_response_text, original_language=language, message_metadata=final_metadata
                     )

                # Save assistant message (either generated or fallback)
                db.add(assistant_message)

                # Update conversation title if it's the default and update timestamp
                # Fetch conversation object again within session to update it safely
                conversation_to_update = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
                if conversation_to_update:
                    # Check if it's the first user message effectively by checking default title
                    if conversation_to_update.title == "New Conversation" or conversation_to_update.title == "New Web Chat Session":
                        # Use the user's first message content for the title
                        title = content[:50] + "..." if len(content) > 50 else content
                        conversation_to_update.title = title
                        logger.info(f"WS: Updated conversation {conv_uuid} title to '{title}'")
                    conversation_to_update.updated_at = datetime.utcnow()
                else:
                    logger.error(f"WS: Could not find conversation {conv_uuid} to update metadata.")

                db.commit() # Commit message saving and conversation update

                # Send final completion message only if stream was successful
                if stream_successful:
                    await websocket.send_json({"type": "complete", "message_id": str(assistant_message_id), "content": full_response_text, "metadata": final_metadata})

                end_time = time.time()
                logger.info(f"WS: Message processed successfully in {end_time - start_time:.2f}s for conv {conversation_id}")

            except Exception as processing_err:
                db.rollback()
                logger.error(f"WS: Error during message processing logic for conv {conversation_id}: {processing_err}", exc_info=True)
                await FallbackHandler.log_fallback(content, processing_err, str(conv_uuid))
                try: await websocket.send_json({"type": "error", "message": "An error occurred while processing your message."})
                except Exception: pass
                # Continue the loop even after processing error
                continue

    except WebSocketDisconnect as ws_disc:
        logger.info(f"WS: Client disconnected gracefully (Code: {ws_disc.code}, Reason: {ws_disc.reason}) for conversation {conversation_id} user {current_user.email if current_user else 'Unknown'}")
    except Exception as loop_err:
        logger.error(f"WS: Unexpected error in connection loop for conv {conversation_id}: {loop_err}", exc_info=True)
        try:
            # Attempt to close gracefully if possible
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass # Ignore errors during close
    finally:
        ws_manager.disconnect(websocket, conversation_id)
        db.close() # Close the dedicated DB session for this connection
        logger.info(f"WS: Cleaned up connection resources for conversation {conversation_id}")


# ===================================
# --- Admin Panel Router (Session Auth) ---
# ===================================

# --- Admin Login/Logout Routes (NO session dependency here) ---
admin_auth_router = APIRouter(prefix="/admin", tags=["Admin Auth"], include_in_schema=False)

@admin_auth_router.get("/login", response_class=HTMLResponse, name="admin_login_page")
async def admin_login_page(request: FastAPIRequest, error: Optional[str] = None):
    """Serves the admin login page."""
    if request.session.get("is_admin_logged_in"):
        return RedirectResponse(request.url_for('admin_dashboard'), status_code=status.HTTP_303_SEE_OTHER)
    # Ensure the template exists
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": error})

@admin_auth_router.post("/login")
async def process_admin_login(
    request: FastAPIRequest, # Inject request to access session
    db: Session = Depends(get_db),
    username: str = Form(...), # Get from form
    password: str = Form(...)  # Get from form
):
    """Processes the admin login form submission."""
    login_url_obj = request.url_for('admin_login_page') # For redirecting back on failure
    user = db.query(User).filter(User.email == username).first()

    if not user or not user.is_active or not user.is_admin or not verify_password(password, user.hashed_password):
        logger.warning(f"Admin login failed for email: {username}")
        error_msg = "Invalid credentials or not an admin account."
        # Pass error message as query parameter for the template to display
        redirect_url = login_url_obj.include_query_params(error=error_msg)
        return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER) # Use 303 and pass error

    # Set session variables upon successful login
    request.session["is_admin_logged_in"] = True
    request.session["admin_user_email"] = user.email
    request.session["admin_user_id"] = str(user.id) # Store as string
    logger.info(f"Admin login successful for user: {user.email}")

    dashboard_url = request.url_for('admin_dashboard')
    return RedirectResponse(dashboard_url, status_code=status.HTTP_303_SEE_OTHER) # Use 303

@admin_auth_router.get("/logout", name="admin_logout")
async def process_admin_logout(request: FastAPIRequest):
    """Logs out the admin by clearing session data."""
    admin_email = request.session.get("admin_user_email")
    request.session.clear() # Clear the entire session for admin logout
    logger.info(f"Admin logout successful for user: {admin_email}")
    login_url = request.url_for('admin_login_page')
    return RedirectResponse(login_url, status_code=status.HTTP_303_SEE_OTHER)

# Include the auth router *without* the main admin dependency
app.include_router(admin_auth_router)


# --- Admin Panel Main Routes ---
# Apply session verification dependency to ALL routes in this router
admin_router = APIRouter(
    prefix="/admin",
    tags=["Admin Panel"],
    dependencies=[Depends(verify_admin_session)], # Apply session check here
    include_in_schema=False # Hide admin routes from OpenAPI docs
)

# The verify_admin_session dependency handles the redirect if not logged in

@admin_router.get("/", response_class=HTMLResponse, name="admin_dashboard")
async def admin_dashboard(request: FastAPIRequest, db: Session = Depends(get_db)):
    # If we reach here, verify_admin_session passed
    try:
        # Fetch analytics data
        analytics_data = {
            "total_users": db_manager.get_total_users_count(db),
            "total_conversations": db_manager.get_total_conversations_count(db),
            "active_conversations": db_manager.get_active_conversations_count(db),
            "total_messages": db_manager.get_total_messages_count(db),
            "messages_today": db_manager.get_messages_today_count(db),
            "uploaded_docs": db_manager.get_uploaded_documents_count(db),
            "total_curated": db_manager.get_curated_responses_count(db),
            "active_curated": db_manager.get_active_curated_responses_count(db),
        }
        logger.info("Fetched analytics data for admin dashboard.")
    except Exception as e:
        logger.error(f"Error fetching analytics for admin dashboard: {e}", exc_info=True)
        analytics_data = {"error": "Could not load analytics data."}

    context = {"request": request, "analytics": analytics_data}
    # Ensure the template exists
    return templates.TemplateResponse("admin_dashboard.html", context)

@admin_router.get("/kb", response_class=HTMLResponse, name="admin_kb_list")
async def admin_kb_list(request: FastAPIRequest, db: Session = Depends(get_db)):
    """Displays the list of uploaded document tracking records."""
    try:
        documents = db_manager.list_uploaded_documents(db)
        return templates.TemplateResponse("admin_kb.html", {"request": request, "documents": documents})
    except Exception as e:
        logger.error(f"Admin KB List Error: {e}", exc_info=True)
        # Render template with error message
        return templates.TemplateResponse("admin_kb.html", {"request": request, "documents": [], "error": "Could not load documents"})

@admin_router.post("/kb/upload")
async def admin_kb_upload(request: FastAPIRequest, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Handles admin uploading a document for knowledge base ingestion."""
    user_id_str = request.session.get("admin_user_id") # Get admin user ID from session
    if not user_id_str:
        # This shouldn't happen due to verify_admin_session, but defensive check
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin session invalid")
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid admin user ID in session")

    # Base URL for redirection
    base_redirect_url_obj = request.url_for('admin_kb_list')

    if not file or not file.filename:
        redirect_url = base_redirect_url_obj.include_query_params(upload_error="nofile")
        return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)

    # Define allowed file types for admin upload
    allowed_types = ["application/pdf", "text/plain", "text/csv", "application/json"]
    if file.content_type not in allowed_types:
        logger.warning(f"Admin Upload rejected: Unsupported file type '{file.content_type}'")
        error_msg = f"Unsupported type: {file.content_type}. Allowed: PDF, TXT, CSV, JSON."
        redirect_url = base_redirect_url_obj.include_query_params(upload_error=error_msg)
        return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)


    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    # Sanitize filename slightly (more robust sanitization might be needed)
    safe_filename_base = "".join(c for c in os.path.basename(file.filename) if c.isalnum() or c in ('-', '_', '.'))
    safe_filename = f"{uuid.uuid4()}_{safe_filename_base}"
    temp_file_path = os.path.join(temp_dir, safe_filename)
    query_params = {} # Initialize query params dict

    try:
        # Save uploaded file temporarily
        async with aiofiles.open(temp_file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024): # Read in chunks
                await buffer.write(content)
        logger.info(f"Admin Upload: File saved temporarily to: {temp_file_path}")

        # Process and ingest the file (run blocking DB call in thread)
        logger.info(f"Admin Upload: Starting ingestion process for {temp_file_path}")
        start_ingest = time.time()
        added_ids = await asyncio.to_thread(db_manager.add_data_from_file, temp_file_path)
        end_ingest = time.time()
        logger.info(f"Admin Upload: Ingestion completed in {end_ingest - start_ingest:.2f} seconds. Added {len(added_ids)} vector chunks.")

        if not added_ids:
            # File processed but no useful data extracted or added
            db.rollback() # Ensure no partial transaction state
            query_params = {"upload_error": "nodata"}
            logger.warning(f"Admin Upload: File '{file.filename}' processed but resulted in 0 ingested chunks.")
            redirect_url = base_redirect_url_obj.include_query_params(**query_params)
            return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)

        # Add tracking record and commit ONLY if ingestion was successful
        db_manager.add_uploaded_document_record(filename=file.filename, user_id=user_id, db=db)
        db.commit()
        logger.info(f"Admin Upload: Tracking record added and ingestion committed for {file.filename}")
        query_params = {"upload_success": "true"}

    except Exception as e:
        db.rollback() # Rollback any potential changes (like tracking record if added before error)
        logger.error(f"Admin Upload Error for {file.filename}: {e}", exc_info=True)
        query_params = {"upload_error": "server"}
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            try:
                # Use await asyncio.to_thread for async cleanup
                await asyncio.to_thread(os.remove, temp_file_path)
                logger.info(f"Admin Upload: Removed temporary file: {temp_file_path}")
            except Exception as e_remove:
                logger.error(f"Admin Upload: Failed to remove temp file {temp_file_path}: {e_remove}")

    # Construct final redirect URL with collected query parameters
    final_redirect_url = base_redirect_url_obj.include_query_params(**query_params)
    return RedirectResponse(final_redirect_url, status_code=status.HTTP_303_SEE_OTHER)


@admin_router.post("/kb/{doc_id}/delete_record")
async def admin_kb_delete_record(request: FastAPIRequest, doc_id: uuid.UUID, db: Session = Depends(get_db)):
    """Deletes the tracking record for an uploaded document (does not delete vectors)."""
    user_id_str = request.session.get("admin_user_id")
    user_id = uuid.UUID(user_id_str) # For logging/consistency
    base_redirect_url_obj = request.url_for('admin_kb_list')
    query_params = {}
    try:
        deleted = db_manager.delete_uploaded_document_record(doc_id=doc_id, user_id=user_id, db=db)
        if deleted:
            db.commit()
            logger.info(f"Admin deleted document tracking record {doc_id}")
            query_params = {"delete_success": "true"}
        else:
            db.rollback()
            logger.warning(f"Admin attempted to delete non-existent tracking record {doc_id}")
            query_params = {"delete_error": "notfound"}
    except Exception as e:
        db.rollback()
        logger.error(f"Admin Error deleting tracking record {doc_id}: {e}", exc_info=True)
        query_params = {"delete_error": "server"}

    final_redirect_url = base_redirect_url_obj.include_query_params(**query_params)
    return RedirectResponse(final_redirect_url, status_code=status.HTTP_303_SEE_OTHER)

@admin_router.get("/curation", response_class=HTMLResponse, name="admin_curation_list")
async def admin_curation_list(request: FastAPIRequest, db: Session = Depends(get_db)):
    """Displays the list of curated responses."""
    try:
        responses = db_manager.list_curated_responses(db)
        return templates.TemplateResponse("admin_curation.html", {"request": request, "responses": responses})
    except Exception as e:
        logger.error(f"Admin Curation List Error: {e}", exc_info=True)
        return templates.TemplateResponse("admin_curation.html", {"request": request, "responses": [], "error": "Could not load curated responses"})

@admin_router.post("/curation/add")
async def admin_curation_add(request: FastAPIRequest, query_pattern: str = Form(...), response: str = Form(...), db: Session = Depends(get_db)):
    """Adds a new curated query-response pair."""
    user_id_str = request.session.get("admin_user_id")
    user_id = uuid.UUID(user_id_str)
    base_redirect_url_obj = request.url_for('admin_curation_list')
    query_params = {}

    if not query_pattern or not response:
        query_params = {"add_error": "empty"}
        redirect_url = base_redirect_url_obj.include_query_params(**query_params)
        return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    try:
        # Add curated response (checks for duplicates inside db_manager)
        db_manager.add_curated_response(query_pattern.strip(), response.strip(), user_id, db)
        db.commit()
        logger.info(f"Admin {user_id} added curated response for pattern: '{query_pattern[:50]}...'")
        query_params = {"add_success": "true"}
    except ValueError as ve: # Specific error for existing pattern from db_manager
        db.rollback()
        logger.warning(f"Admin Curation Add Failed: {ve}")
        query_params = {"add_error": "exists"}
    except Exception as e:
        db.rollback()
        logger.error(f"Admin Curation Add Error: {e}", exc_info=True)
        query_params = {"add_error": "server"}

    final_redirect_url = base_redirect_url_obj.include_query_params(**query_params)
    return RedirectResponse(final_redirect_url, status_code=status.HTTP_303_SEE_OTHER)

@admin_router.post("/curation/{resp_id}/toggle")
async def admin_curation_toggle(request: FastAPIRequest, resp_id: uuid.UUID, db: Session = Depends(get_db)):
    """Toggles the active status of a curated response."""
    base_redirect_url_obj = request.url_for('admin_curation_list')
    query_params = {}
    try:
        updated_resp = db_manager.toggle_curated_response_status(response_id=resp_id, db=db)
        if updated_resp:
            db.commit()
            logger.info(f"Admin toggled curated response {resp_id} status to {updated_resp.is_active}")
            query_params = {"toggle_success": "true"}
        else:
            db.rollback()
            query_params = {"toggle_error": "notfound"}
    except Exception as e:
        db.rollback()
        logger.error(f"Admin Curation Toggle Error for {resp_id}: {e}", exc_info=True)
        query_params = {"toggle_error": "server"}

    final_redirect_url = base_redirect_url_obj.include_query_params(**query_params)
    return RedirectResponse(final_redirect_url, status_code=status.HTTP_303_SEE_OTHER)

@admin_router.post("/curation/{resp_id}/delete")
async def admin_curation_delete(request: FastAPIRequest, resp_id: uuid.UUID, db: Session = Depends(get_db)):
    """Deletes a curated response."""
    base_redirect_url_obj = request.url_for('admin_curation_list')
    query_params = {}
    try:
        deleted = db_manager.delete_curated_response(response_id=resp_id, db=db)
        if deleted:
            db.commit()
            logger.info(f"Admin deleted curated response {resp_id}")
            query_params = {"delete_success": "true"}
        else:
            db.rollback()
            query_params = {"delete_error": "notfound"}
    except Exception as e:
        db.rollback()
        logger.error(f"Admin Curation Delete Error for {resp_id}: {e}", exc_info=True)
        query_params = {"delete_error": "server"}

    final_redirect_url = base_redirect_url_obj.include_query_params(**query_params)
    return RedirectResponse(final_redirect_url, status_code=status.HTTP_303_SEE_OTHER)

# --- Include the main Admin Router (which has the session dependency) ---
app.include_router(admin_router)

# ===================================
# --- Other General Endpoints ---
# ===================================
@app.get("/api/health", tags=["Health"])
async def health_check():
    """Provides a health check endpoint for monitoring."""
    db_ok = False
    try:
        # Use context manager for connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:
        logger.warning(f"Health check DB connection failed: {e}")
        db_ok = False

    return {
        "status": "ok",
        "database_connected": db_ok,
        "model_loaded": model_manager.is_model_loaded(),
        "timestamp": datetime.utcnow().isoformat(),
        "memory": MemoryMonitor.get_memory_usage() # Include memory usage
    }

# --- Language API Removed ---

# --- User Document Upload API Removed ---
# The /api/upload endpoint is now only used by the admin panel (/admin/kb/upload).
# If user uploads are ever needed again, a new route would be required.

# --- Serve Frontend Route ---
@app.get("/", include_in_schema=False)
async def read_index(request: Request):
    """Serves the main index.html frontend."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(root_dir, "index.html")
    if not os.path.exists(index_path):
        logger.error("index.html not found!")
        raise HTTPException(status_code=404, detail="Frontend application not found.")
    return FileResponse(index_path)

# Serve static files (CSS, JS) if needed (place them in a 'static' folder)
# app.mount("/static", StaticFiles(directory="static"), name="static")


# ===================================
# --- Main Execution ---
# ===================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    # Check UVICORN_RELOAD environment variable
    reload_env = os.getenv("UVICORN_RELOAD", "false").lower()
    reload = reload_env == "true" or reload_env == "1"
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")

    if reload:
        logger.warning("Uvicorn reload is ENABLED. Do not use in production.")

    logger.info(f"Starting Uvicorn server on http://{host}:{port} with reload={reload}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )