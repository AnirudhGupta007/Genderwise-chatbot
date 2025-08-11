GenderWise Chatbot
==================

GenderWise is a sophisticated RAG (Retrieval Augmented Generation) chatbot designed to provide accurate and concise information on gender issues, with a primary focus on India. This project leverages an API-based LLM for generation, a Sentence Transformers model for creating text embeddings, and PostgreSQL with the pgvector extension for efficient semantic search and retrieval.

The backend is built with FastAPI and includes a complete user authentication system and a powerful admin panel for managing the knowledge base and curating responses.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ![alt text]()   `

Table of Contents
-----------------

*   [Project Overview](https://www.google.com/url?sa=E&q=#project-overview)
    
*   [Features](https://www.google.com/url?sa=E&q=#features)
    
*   [Tech Stack](https://www.google.com/url?sa=E&q=#tech-stack)
    
*   [Architecture](https://www.google.com/url?sa=E&q=#architecture)
    
*   [Setup and Installation](https://www.google.com/url?sa=E&q=#setup-and-installation)
    
*   [Environment Variables](https://www.google.com/url?sa=E&q=#environment-variables)
    
*   [Running the Application](https://www.google.com/url?sa=E&q=#running-the-application)
    
*   [Data Ingestion](https://www.google.com/url?sa=E&q=#data-ingestion)
    
*   [API Endpoints](https://www.google.com/url?sa=E&q=#api-endpoints)
    
*   [Future Improvements](https://www.google.com/url?sa=E&q=#future-improvements)
    

Project Overview
----------------

This project was developed to create an intelligent and reliable source of information on a critical subject. It follows a RAG architecture to ground LLM responses in factual data from a managed knowledge base, ensuring relevance and accuracy. The system is designed for scalability and manageability, having transitioned from a locally-hosted fine-tuned Llama model to a more flexible API-based approach.

Features
--------

*   **Retrieval Augmented Generation (RAG):** Provides contextually relevant answers by retrieving information from a vector database before generation.
    
*   **API-Based LLM:** Utilizes powerful models like OpenAI's GPT-4o for high-quality, low-latency responses, simplifying maintenance and infrastructure requirements.
    
*   **Real-time Chat:** A dynamic frontend communicates with the backend via WebSockets for a seamless, streaming chat experience.
    
*   **User Authentication:** Secure JWT-based registration and login system, including Google OAuth for easy sign-on.
    
*   **Admin Panel:** A comprehensive web interface for administrators to:
    
    *   View system analytics (user counts, message volumes, etc.).
        
    *   Manage the knowledge base by uploading new documents.
        
    *   Create and manage curated responses for specific queries, bypassing the RAG pipeline.
        
*   **Batch Data Ingestion:** Includes a command-line script for bulk-uploading and processing documents into the knowledge base.
    

Tech Stack
----------

**ComponentTechnology / LibraryBackendPython 3.10+, FastAPIWeb Server**Uvicorn, Gunicorn**DatabasePostgreSQL (v16+)Vector Storepgvector** (PostgreSQL Extension)**ORM**SQLAlchemy 2.0**LLM IntegrationOpenAI API** (or other API-based LLMs)**EmbeddingsSentence Transformers** (all-MiniLM-L6-v2)**Authentication**JWT (python-jose), Passlib (bcrypt), Google OAuth (Authlib)**Real-time Comms**WebSockets**Admin Frontend**Jinja2 Templates**DeploymentDocker**, Docker Compose

Architecture
------------

The application is structured around a robust backend that serves both the user-facing chat application and the admin panel.

1.  **User Interaction:** The user interacts with a single-page frontend (index.html), which establishes a WebSocket connection for a given conversation.
    
2.  **Authentication:** Users register/log in via REST API endpoints, receiving a JWT token which is then used to authenticate WebSocket connections.
    
3.  **RAG Pipeline:**
    
    *   A user query is received via WebSocket.
        
    *   The system checks for a matching **curated response**. If found, it's returned immediately.
        
    *   Otherwise, the query is embedded using **Sentence Transformers**.
        
    *   The embedding is used to perform a similarity search in the **pgvector** database to find relevant document chunks.
        
    *   The retrieved chunks, chat history, and the user's query are formatted into a prompt.
        
4.  **Generation:** The prompt is sent to the **OpenAI API**. The streamed response is relayed back to the user through the WebSocket.
    
5.  **Data Persistence:** All conversations and messages are stored in the PostgreSQL relational database. Document metadata and curated responses are also managed here.
    

Setup and Installation
----------------------

### Prerequisites

*   Python 3.10+
    
*   PostgreSQL (v12+ recommended) with the pgvector extension enabled.
    
*   Git
    
*   Docker & Docker Compose (for containerized deployment)
    

### Local Setup Steps

1.  codeBashgit clone https://github.com/your-username/genderwise-chatbot.gitcd genderwise-chatbot
    
2.  codeBashpython3 -m venv venvsource venv/bin/activate
    
3.  codeBashpip install -r requirements.txt
    
4.  **Set up PostgreSQL:**
    
    *   Ensure your local PostgreSQL server is running.
        
    *   Create a user and a database.
        
    *   Connect to your database and enable the pgvector extension: CREATE EXTENSION IF NOT EXISTS vector;
        
5.  **Configure Environment Variables:**
    
    *   codeBashcp .env.example .env
        
    *   Edit the .env file with your specific configurations (Database URL, OpenAI API Key, etc.). See the section below for details.
        

Environment Variables
---------------------

Create a .env file in the project root and populate it with the following variables:

codeDotenv

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # .env  # --- Database ---  DATABASE_URL="postgresql://chatbot_user:your_password@localhost:5432/chatbot_db"  # --- LLM API Configuration ---  LLM_API_PROVIDER="OPENAI"  OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  API_LLM_MODEL_NAME="gpt-4o"  # --- RAG & Embeddings ---  EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"  TEXT_CHUNK_SIZE=500  TEXT_CHUNK_OVERLAP=100  RETRIEVER_K=3  # --- Authentication ---  SECRET_KEY="generate_a_strong_random_key_using_openssl_rand_hex_32"  ALGORITHM="HS256"  ACCESS_TOKEN_EXPIRE_MINUTES=1440 # 24 hours  # --- Google OAuth (Optional) ---  GOOGLE_CLIENT_ID="your_google_client_id.apps.googleusercontent.com"  GOOGLE_CLIENT_SECRET="your_google_client_secret"  # --- Web Server ---  ALLOWED_ORIGINS="http://localhost:8000,http://127.0.0.1:8000"  HOST="127.0.0.1"  PORT=8000  # --- Tokenizers Parallelism (to suppress warnings) ---  TOKENIZERS_PARALELLELISM=false   `

Running the Application
-----------------------

### Using Uvicorn (for local development)

1.  codeBashuvicorn main:app --host 0.0.0.0 --port 8000 --reload
    
2.  Open your browser and navigate to http://localhost:8000.
    
3.  Log in to the admin panel at http://localhost:8000/admin/login.
    

### Using Docker

1.  Ensure Docker is running.
    
2.  codeBashdocker build -t genderwise-chatbot:latest .
    
3.  codeBashdocker run -d --name genderwise-app -p 8000:8000 --env-file .env genderwise-chatbot:latest(See docker-compose.yml for a more managed setup, including volumes for persistent logs).
    

Data Ingestion
--------------

You can populate the knowledge base in two ways:

1.  **Admin Panel:** Log in to the admin panel, navigate to "Knowledge Base," and upload individual files (.pdf, .txt, .json, .csv).
    
2.  codeBashpython ingest\_batch.py -d "/path/to/your/documents" -u "your\_admin\_email@example.com" --move
    

API Endpoints
-------------

The application exposes several REST API endpoints, primarily for authentication and conversation management. The core chat functionality is handled via WebSockets.

*   /api/auth/register: User registration.
    
*   /api/auth/token: User login, returns JWT token.
    
*   /api/auth/login/google: Initiates Google OAuth flow.
    
*   /api/conversations: (Protected) CRUD operations for user conversations.
    
*   /ws/chat/{conversation\_id}: WebSocket endpoint for real-time chat.
    
*   /api/health: Health check endpoint.
    

Future Improvements
-------------------

*   Implement a more robust method for deleting vectors from the knowledge base.
    
*   Add unit and integration tests for key application logic.
    
*   Enhance the RAG pipeline with techniques like re-ranking or query expansion.
    
*   Develop a more sophisticated frontend interface using a modern framework like React or Vue.
    

**How to customize it:**

*   **\[Chatbot Screenshot (Placeholder)\]():** Take a nice screenshot of your application's chat interface and add it here. You can upload the image to your GitHub repo and link to it.
    
*   **git clone https://github.com/your-username/genderwise-chatbot.git:** Replace this with the actual URL of your GitHub repository.
    
*   **.env.example:** It's good practice to create a file named .env.example in your repo that has all the variable names but no secret values, and mention it in the README.
    
*   **License:** Consider adding a LICENSE file (e.g., MIT, Apache 2.0) to your repository to clarify how others can use your code. GitHub provides an easy way to add one.
