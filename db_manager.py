# db_manager.py

import os
import logging
import uuid
from datetime import datetime, date, timezone
# Keep existing sqlalchemy imports
from sqlalchemy import create_engine, text, inspect, select, delete, update, func
from sqlalchemy.orm import sessionmaker, Session
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import pandas as pd
import json
# --- Corrected typing import ---
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
# ------------------------------
import hashlib
from dotenv import load_dotenv
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
    logging.warning("pypdf library not found. PDF processing will be skipped. Run 'pip install pypdf'")

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/chatbot_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME", "document_embeddings")
CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "200"))

if torch.backends.mps.is_available(): 
    DEVICE = "mps"
elif torch.cuda.is_available():       
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

logger.info(f"Using device: {DEVICE} for embeddings")

try:
    # Ensure pool_pre_ping is useful for potential connection drops
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
except Exception as e:
    logger.critical(f"Failed to create database engine: {e}")
    raise

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# --- TYPE CHECKING Block ---
# This block is only processed by type checkers (like Pylance), not at runtime.
# It allows us to import types for hinting without causing circular import errors.
if TYPE_CHECKING:
    from main import UploadedDocument, CuratedResponse, User, Conversation, Message
# ---------------------------


class DBManager:
    """Database manager for handling document ingestion and vector store management"""

    def __init__(self):
        self.engine = engine
        self.collection_name = COLLECTION_NAME
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": DEVICE}
            )
            self.embedding_dimensions = self._get_embedding_dimensions()
        except Exception as e:
            logger.critical(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
            raise

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.vector_store = self._initialize_vector_store()

    def _get_embedding_dimensions(self) -> int:
        """Determine the embedding dimension from the loaded model."""
        try:
            # Use a simple, common word for embedding test
            dummy_embedding = self.embedding_model.embed_query("test")
            if not isinstance(dummy_embedding, list) or not all(isinstance(x, float) for x in dummy_embedding):
                 raise TypeError("Embedding function did not return a list of floats.")
            dim = len(dummy_embedding)
            logger.info(f"Determined embedding dimension: {dim}")
            return dim
        except Exception as e:
            logger.error(f"Could not determine embedding dimension, falling back to default 384. Error: {e}")
            # Fallback dimension, ensure it matches your model if possible
            return 384

    def _initialize_vector_store(self) -> PGVector:
        """Initialize vector store connection, setting up pgvector if needed."""
        try:
            logger.info(f"Initializing PGVector store with collection: {self.collection_name}")
            self._setup_pgvector() # Ensures extension and tables are ready
            vector_store = PGVector(
                connection_string=DATABASE_URL,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                # pre_delete_collection=False # Default is False, keep data across restarts
            )
            logger.info("Vector store initialized successfully")
            return vector_store
        except Exception as e:
            logger.critical(f"FATAL: Error initializing vector store: {e}", exc_info=True)
            raise

    def _setup_pgvector(self):
        """Set up pgvector extension and Langchain tables if they don't exist."""
        try:
            with self.engine.connect() as connection:
                # 1. Ensure pgvector extension exists
                logger.info("Checking/Creating pgvector extension...")
                trans = connection.begin() # Use transaction
                try:
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    trans.commit()
                    logger.info("pgvector extension ensured.")
                except Exception as ext_err:
                    trans.rollback()
                    logger.error(f"Error creating vector extension (might require DB superuser privileges): {ext_err}")
                    # Decide if this is fatal or can be ignored if extension already exists
                    # Re-raising might be appropriate if it's the first setup
                    raise

                # 2. Check and create Langchain metadata tables within a transaction
                inspector = inspect(self.engine)
                trans = connection.begin()
                try:
                    if not inspector.has_table("langchain_pg_collection"):
                        logger.info("Creating table: langchain_pg_collection")
                        connection.execute(text("""
                        CREATE TABLE langchain_pg_collection (
                            uuid UUID PRIMARY KEY,
                            name VARCHAR,
                            cmetadata JSON
                        )
                        """))
                        # Add unique constraint on name for safety
                        connection.execute(text("ALTER TABLE langchain_pg_collection ADD CONSTRAINT name_unique UNIQUE (name);"))
                    else:
                        logger.info("Table 'langchain_pg_collection' already exists.")

                    if not inspector.has_table("langchain_pg_embedding"):
                        logger.info("Creating table: langchain_pg_embedding")
                        # Use f-string carefully for dimension, assuming it's validated integer
                        connection.execute(text(f"""
                        CREATE TABLE langchain_pg_embedding (
                            uuid UUID PRIMARY KEY,
                            collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                            embedding vector({self.embedding_dimensions}),
                            document VARCHAR,
                            cmetadata JSON,
                            custom_id VARCHAR -- Langchain uses this sometimes
                        )
                        """))
                        # Add index (Consider index type based on expected query volume and data size)
                        # IVFFlat is good for speed vs accuracy trade-off. HNSW is another option.
                        # Using vector_cosine_ops as it's common for sentence embeddings.
                        index_name = f"ix_{self.collection_name}_embedding" # Dynamic index name
                        logger.info(f"Creating vector index: {index_name}")
                        connection.execute(text(f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON langchain_pg_embedding
                            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                        """))
                    else:
                        logger.info("Table 'langchain_pg_embedding' already exists.")

                    trans.commit() # Commit table creations together
                    logger.info("Langchain tables checked/created successfully.")

                except Exception as table_err:
                    trans.rollback()
                    logger.error(f"Error during Langchain table setup: {table_err}", exc_info=True)
                    raise # Fail fast if tables can't be set up

        except Exception as e:
            # Catch broader connection errors etc.
            logger.error(f"Error during pgvector/table setup: {e}", exc_info=True)
            raise

    def get_retriever(self, search_type="similarity", search_kwargs=None):
        """Get vector store retriever with specified configuration"""
        if self.vector_store is None:
            logger.error("Vector store not initialized, cannot get retriever.")
            raise RuntimeError("Vector Store is not available.")

        default_k = int(os.getenv("RETRIEVER_K", "5")) # Default K from env or 5
        _search_kwargs = {"k": default_k}
        if search_kwargs: # Allow overriding defaults
            _search_kwargs.update(search_kwargs)

        logger.info(f"Creating retriever with search_type='{search_type}', search_kwargs={_search_kwargs}")
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=_search_kwargs
        )

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """Adds document chunks to the vector store after splitting and hashing."""
        if self.vector_store is None:
            logger.error("Vector store not initialized, cannot add documents.")
            return []
        if not documents:
            logger.warning("add_documents called with an empty list.")
            return []

        processed_docs: List[Document] = []
        logger.info(f"Processing {len(documents)} source documents for chunking...")
        for doc in documents:
            if not isinstance(doc, Document) or not hasattr(doc, 'page_content') or not isinstance(doc.page_content, str):
                logger.warning(f"Skipping invalid document object: {doc}")
                continue
            if not doc.page_content.strip():
                logger.warning(f"Skipping document with empty page_content from source: {doc.metadata.get('source', 'unknown')}")
                continue

            try:
                # Split the document content into chunks
                split_chunks = self.text_splitter.split_documents([doc]) # Pass as list
                for chunk in split_chunks:
                    # Add content hash for potential duplicate checking or identification
                    content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
                    # Ensure metadata exists before adding to it
                    if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                         chunk.metadata = {}
                    chunk.metadata['content_hash'] = content_hash
                    # Preserve original source filename if available
                    chunk.metadata['original_source'] = doc.metadata.get('source', 'unknown')
                    processed_docs.append(chunk)
            except Exception as split_err:
                 logger.error(f"Error splitting document from source {doc.metadata.get('source', 'unknown')}: {split_err}", exc_info=True)
                 continue # Skip this document if splitting fails

        added_ids = []
        total_chunks = len(processed_docs)
        logger.info(f"Prepared {total_chunks} chunks for ingestion.")
        if not processed_docs:
            logger.warning("No processable chunks generated from input documents.")
            return []

        # Use a dedicated session from SessionLocal IF NEEDED for ORM operations within this method.
        # PGVector's add_documents often handles its own connection/transaction.
        # session: Session = SessionLocal() # Only needed if performing ORM ops here
        try:
            for i in range(0, total_chunks, batch_size):
                batch = processed_docs[i : i + batch_size]
                logger.info(f"Adding batch {i // batch_size + 1}/{ (total_chunks + batch_size - 1) // batch_size } ({len(batch)} chunks)...")
                try:
                    ids = self.vector_store.add_documents(batch) # This is the primary operation
                    added_ids.extend(ids)
                    logger.info(f"Batch {i // batch_size + 1} added. {len(ids)} IDs returned.")
                except Exception as batch_err:
                     logger.error(f"Error adding batch {i // batch_size + 1} to vector store: {batch_err}", exc_info=True)
                     # Decide whether to continue with next batch or raise immediately
                     # For robustness, maybe log and continue? Or raise if it's critical.
                     # raise # Uncomment to stop on first batch error
                     continue # Logged error, try next batch

            logger.info(f"Successfully attempted ingestion. Added {len(added_ids)} document chunks in total.")
            return added_ids
        except Exception as e:
            # This catches errors outside the batch loop (less likely)
            logger.error(f"Unexpected error during add_documents loop: {e}", exc_info=True)
            raise # Re-raise unexpected errors
        # finally:
            # session.close() # Only if session was created above


    def add_data_from_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Reads data from various file types and adds it to the vector store."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return []

            file_ext = os.path.splitext(file_path)[1].lower()
            docs_to_add: List[Document] = []
            base_metadata = metadata.copy() if metadata else {} # Ensure base_metadata is a copy
            # Use basename for source metadata to avoid exposing full path
            base_metadata['source'] = os.path.basename(file_path)

            logger.info(f"Processing file: {file_path} (type: {file_ext})")

            if file_ext == '.csv':
                try:
                    df = pd.read_csv(file_path)
                    # Simplified logic: Assume first column is text unless specific keywords found
                    text_col = df.columns[0]
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in ['text', 'content', 'document', 'description', 'abstract', 'body']):
                            text_col = col; break
                    logger.info(f"Using column '{text_col}' as text source for CSV.")
                    for i, row in df.iterrows():
                        page_content = str(row[text_col])
                        if not page_content.strip(): continue
                        row_meta = {col: str(row[col]) for col in df.columns if col != text_col} # Convert all metadata to string? Be careful.
                        row_meta.update(base_metadata)
                        row_meta['row_id'] = i # Add row number
                        docs_to_add.append(Document(page_content=page_content, metadata=row_meta))
                except Exception as csv_err:
                    logger.error(f"Error processing CSV file {file_path}: {csv_err}", exc_info=True)
                    return [] # Stop processing this file on error

            elif file_ext == '.json':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
                    # Handle list of objects/strings or single object
                    items_to_process = []
                    if isinstance(data, list): items_to_process = data
                    elif isinstance(data, dict): items_to_process = [data] # Treat single dict as a list of one
                    else: logger.warning(f"Unsupported JSON structure in {file_path} (expected list or dict)."); return []

                    for i, item in enumerate(items_to_process):
                        item_meta = base_metadata.copy()
                        item_meta['item_id'] = i # Add item index
                        page_content = ""
                        if isinstance(item, dict):
                            text_val = None
                            for key, value in item.items():
                                # Simple text extraction heuristic
                                if any(keyword in key.lower() for keyword in ['text', 'content', 'document', 'description', 'abstract', 'body']) and isinstance(value, str):
                                    text_val = value
                                    # Optionally add other fields to metadata (ensure they are serializable)
                                    try: json.dumps({key: value}); item_meta[key] = value
                                    except TypeError: logger.debug(f"Skipping non-serializable metadata key '{key}' in JSON item {i}")
                                else:
                                    try: json.dumps({key: value}); item_meta[key] = value
                                    except TypeError: logger.debug(f"Skipping non-serializable metadata key '{key}' in JSON item {i}")

                            if text_val: page_content = text_val
                            else: page_content = json.dumps(item) # Fallback: stringify whole item
                        elif isinstance(item, str): page_content = item
                        else: page_content = str(item) # Fallback: convert other types

                        if page_content.strip():
                            docs_to_add.append(Document(page_content=page_content, metadata=item_meta))
                except json.JSONDecodeError as json_err:
                     logger.error(f"Error decoding JSON file {file_path}: {json_err}", exc_info=True)
                     return []
                except Exception as json_proc_err:
                     logger.error(f"Error processing JSON data from {file_path}: {json_proc_err}", exc_info=True)
                     return []

            elif file_ext in ['.txt', '.md']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
                    # Treat whole file as one document, splitting happens in add_documents
                    if text.strip():
                        para_meta = base_metadata.copy()
                        # Optionally split by paragraphs here if preferred over text_splitter
                        # paragraphs = text.split('\n\n')
                        # for i, para in enumerate(paragraphs):
                        #     if para.strip():
                        #         para_meta = base_metadata.copy(); para_meta['paragraph'] = i
                        #         docs_to_add.append(Document(page_content=para.strip(), metadata=para_meta))
                        docs_to_add.append(Document(page_content=text.strip(), metadata=para_meta))
                except Exception as text_err:
                     logger.error(f"Error reading text file {file_path}: {text_err}", exc_info=True)
                     return []

            elif file_ext == '.pdf':
                if PdfReader is None:
                     logger.error("Cannot process PDF, pypdf library is not installed.")
                     return []
                try:
                    with open(file_path, 'rb') as f:
                        reader = PdfReader(f)
                        num_pages = len(reader.pages)
                        logger.info(f"Reading PDF '{os.path.basename(file_path)}' with {num_pages} pages.")
                        for i, page in enumerate(reader.pages):
                            try:
                                page_content = page.extract_text()
                                if page_content and page_content.strip():
                                    page_meta = base_metadata.copy()
                                    page_meta['page_number'] = i + 1
                                    page_meta['total_pages'] = num_pages
                                    docs_to_add.append(Document(page_content=page_content.strip(), metadata=page_meta))
                                else:
                                    logger.warning(f"No text extracted from page {i+1} of {os.path.basename(file_path)}.")
                            except Exception as page_err:
                                logger.error(f"Error extracting text from page {i+1} of {os.path.basename(file_path)}: {page_err}")
                                continue # Try next page
                except Exception as pdf_err:
                    logger.error(f"Error reading PDF file {file_path}: {pdf_err}", exc_info=True)
                    return []

            elif file_ext == '.docx':
                logger.warning(f"DOCX processing not implemented yet. Install 'python-docx' and add logic.") # Placeholder
                # Example structure (requires pip install python-docx)
                # try:
                #     from docx import Document as DocxDocument
                #     document = DocxDocument(file_path)
                #     full_text = "\n".join([para.text for para in document.paragraphs])
                #     if full_text.strip():
                #         docx_meta = base_metadata.copy()
                #         docs_to_add.append(Document(page_content=full_text.strip(), metadata=docx_meta))
                # except ImportError:
                #     logger.error("python-docx library not found. Cannot process DOCX.")
                # except Exception as docx_err:
                #     logger.error(f"Error processing DOCX file {file_path}: {docx_err}", exc_info=True)
                return [] # Not implemented
            else:
                logger.warning(f"Unsupported file type: {file_ext} for file {file_path}")
                return [] # Skip unsupported types

            # Final step: Add extracted documents (if any) to the vector store
            if docs_to_add:
                logger.info(f"Submitting {len(docs_to_add)} processed documents from {os.path.basename(file_path)} for ingestion.")
                # Pass through add_documents for final chunking and vectorization
                return self.add_documents(docs_to_add)
            else:
                logger.warning(f"No processable documents generated from file: {file_path}")
                return []
        except Exception as e:
            # Catch-all for errors during file access or initial processing
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            # Depending on desired behavior, you might want to re-raise
            # raise # Uncomment if errors during file processing should halt the caller
            return [] # Or return empty list on error

    # --- Admin Panel DB Functions ---
    # Using string literals for type hints involving models from main.py

    def add_uploaded_document_record(self, filename: str, user_id: uuid.UUID, db: Session) -> 'UploadedDocument':
        """Adds a tracking record for an uploaded document."""
        from main import UploadedDocument # Local import for runtime
        new_record = UploadedDocument(filename=filename, user_id=user_id)
        db.add(new_record)
        # No commit here, let the calling route handle commit/rollback
        return new_record

    def list_uploaded_documents(self, db: Session) -> List['UploadedDocument']:
        """Lists uploaded document tracking records."""
        from main import UploadedDocument # Local import for runtime
        return db.query(UploadedDocument).order_by(UploadedDocument.uploaded_at.desc()).all()

    def delete_uploaded_document_record(self, doc_id: uuid.UUID, user_id: uuid.UUID, db: Session) -> bool:
        """Deletes a tracking record (does not delete vectors)."""
        from main import UploadedDocument # Local import for runtime
        result = db.execute(
            delete(UploadedDocument).where(UploadedDocument.id == doc_id)
            # Optional: Add .where(UploadedDocument.user_id == user_id) if only uploader can delete
        )
        # No commit here
        return result.rowcount > 0

    def add_curated_response(self, query_pattern: str, response: str, user_id: uuid.UUID, db: Session) -> 'CuratedResponse':
        """Adds a new curated response."""
        from main import CuratedResponse # Local import for runtime
        exists = db.query(CuratedResponse.id).filter(CuratedResponse.query_pattern == query_pattern).first()
        if exists:
            raise ValueError("Query pattern already exists.")

        new_resp = CuratedResponse(
            query_pattern=query_pattern,
            response=response,
            user_id=user_id,
            is_active=True # Default to active
        )
        db.add(new_resp)
        # No commit here
        return new_resp

    def list_curated_responses(self, db: Session) -> List['CuratedResponse']:
        """Lists all curated responses."""
        from main import CuratedResponse # Local import for runtime
        return db.query(CuratedResponse).order_by(CuratedResponse.created_at.desc()).all()

    def get_active_curated_response(self, query_pattern: str, db: Session) -> Optional['CuratedResponse']:
        """Gets an active curated response matching the exact query pattern."""
        from main import CuratedResponse # Local import for runtime
        return db.query(CuratedResponse).filter(
            CuratedResponse.query_pattern == query_pattern,
            CuratedResponse.is_active == True
        ).first()

    def toggle_curated_response_status(self, response_id: uuid.UUID, db: Session) -> Optional['CuratedResponse']:
        """Toggles the active status of a curated response."""
        from main import CuratedResponse # Local import for runtime
        resp = db.query(CuratedResponse).filter(CuratedResponse.id == response_id).first()
        if resp:
            resp.is_active = not resp.is_active
            # Use timezone aware now if your DB column supports it
            resp.updated_at = datetime.now(datetime.timezone.utc) if resp.updated_at.tzinfo else datetime.utcnow()

            # No commit here
        return resp

    def delete_curated_response(self, response_id: uuid.UUID, db: Session) -> bool:
        """Deletes a curated response."""
        from main import CuratedResponse # Local import for runtime
        result = db.execute(
            delete(CuratedResponse).where(CuratedResponse.id == response_id)
        )
        # No commit here
        return result.rowcount > 0

    # --- Analytics DB Functions ---

    def get_total_users_count(self, db: Session) -> int:
        """Gets the total number of registered users."""
        from main import User # Local import for runtime
        return db.query(func.count(User.id)).scalar() or 0

    def get_total_conversations_count(self, db: Session) -> int:
        """Gets the total number of conversations (active or inactive)."""
        from main import Conversation # Local import for runtime
        return db.query(func.count(Conversation.id)).scalar() or 0

    def get_active_conversations_count(self, db: Session) -> int:
        """Gets the total number of active conversations."""
        from main import Conversation # Local import for runtime
        return db.query(func.count(Conversation.id)).filter(Conversation.is_active == True).scalar() or 0

    def get_total_messages_count(self, db: Session) -> int:
        """Gets the total number of messages."""
        from main import Message # Local import for runtime
        return db.query(func.count(Message.id)).scalar() or 0

    # In db_manager.py

    def get_messages_today_count(self, db: Session) -> int:
        """Gets the number of messages created today (UTC)."""
        from main import Message # Local import for runtime

        # 1. Get the current date in UTC
        current_utc_date = datetime.now(timezone.utc).date()

        # 2. Create the start of that day (midnight) in UTC
        start_of_day_utc = datetime.combine(current_utc_date, datetime.min.time(), tzinfo=timezone.utc)

        # 3. Filter messages created at or after the start of today UTC
        return db.query(func.count(Message.id)).filter(
            Message.created_at >= start_of_day_utc
        ).scalar() or 0

    def get_uploaded_documents_count(self, db: Session) -> int:
        """Gets the total number of uploaded document tracking records."""
        from main import UploadedDocument # Local import for runtime
        return db.query(func.count(UploadedDocument.id)).scalar() or 0

    def get_curated_responses_count(self, db: Session) -> int:
        """Gets the total number of curated responses."""
        from main import CuratedResponse # Local import for runtime
        return db.query(func.count(CuratedResponse.id)).scalar() or 0

    def get_active_curated_responses_count(self, db: Session) -> int:
        """Gets the total number of active curated responses."""
        from main import CuratedResponse # Local import for runtime
        return db.query(func.count(CuratedResponse.id)).filter(CuratedResponse.is_active == True).scalar() or 0

    # --- Existing document_exists method ---
    def document_exists(self, text: str) -> bool:
        """Checks if a document chunk with the same content hash exists."""
        if self.vector_store is None: return False
        if not text: return False # Cannot hash empty text

        try:
            doc_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            # Use SessionLocal for querying the embedding table if PGVector doesn't provide a direct check
            # Or, if PGVector has a way to filter by metadata, use that.
            # Assuming direct SQL query for now:
            with self.engine.connect() as connection:
                 result = connection.execute(text(f"""
                     SELECT EXISTS (
                         SELECT 1 FROM langchain_pg_embedding
                         WHERE cmetadata->>'content_hash' = :hash
                         AND collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :coll_name LIMIT 1)
                     )
                 """), {"hash": doc_hash, "coll_name": self.collection_name})
                 exists = result.scalar_one_or_none() # Returns True, False, or None
                 return exists is True # Explicitly return boolean
        except Exception as e:
            logger.error(f"Error checking document existence for hash {doc_hash}: {e}", exc_info=True)
            return False # Default to False on error


# --- Singleton Instance ---
# Instantiate this in main.py *after* models are defined
db_manager: Optional[DBManager] = None

# Function to get the singleton instance (Optional, direct import often simpler)
# Ensure this is called only *after* db_manager is initialized in main.py
def get_db_manager():
    """Get singleton DBManager instance. Raises RuntimeError if not initialized."""
    global db_manager
    if db_manager is None:
        # This should ideally not happen if initialization order in main.py is correct
        logger.critical("DBManager accessed before initialization in main.py!")
        raise RuntimeError("DBManager accessed before initialization in main.py")
    return db_manager