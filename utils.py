import os
import logging
import json
import time
import uuid
import asyncio
import gc
from typing import Dict, List, Any, Optional, Union, Tuple,AsyncIterator, Callable
from datetime import datetime
import torch
from langchain_community.llms import LlamaCpp
# Langchain imports remain mostly the same
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import aiofiles
import psutil
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 2 # Number of recent User/Assistant turn pairs to include
FALLBACK_ENABLED = os.getenv("FALLBACK_ENABLED", "true").lower() == "true" # Control fallback

# Configuration from Environment Variables
MODEL_PATH = os.getenv("MODEL_PATH") # Expecting path to .gguf file
if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    logger.critical(f"FATAL: MODEL_PATH environment variable not set or file not found at '{MODEL_PATH}'")
    # Decide how to handle - exit or raise? Raising for clarity.
    raise FileNotFoundError(f"GGUF Model file not found at path specified by MODEL_PATH: {MODEL_PATH}")

# LlamaCpp specific parameters from env
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0")) # Layers to offload to GPU
N_BATCH = int(os.getenv("N_BATCH", "512")) # Batch size for prompt processing
N_CTX = int(os.getenv("N_CTX", "2048")) # Context window size

# LLM Generation Parameters (used by LlamaCpp)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024")) # LlamaCpp uses max_tokens
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
# REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15")) # Check LlamaCpp param name

# Model Management Configuration
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT_SECONDS", "600")) # 10 minutes default
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").lower() == "true"


# utils.py

RAG_PROMPT_TEMPLATE = """You are GenderWise, an AI assistant specialized in gender issues in India.

Provide a concise, focused answer to the user's question. Avoid elaboration unless explicitly requested.

Response Strategy:
1. If the provided context is relevant and sufficient for the user's question about gender issues in India: Use this context to provide a focused answer highlighting critical information.

2. If the context is irrelevant or insufficient, but the question is about gender issues in India: Briefly state that the available information doesn't cover this specific aspect, then provide a concise answer using your general knowledge.

3. If the question is not about gender issues in India: Politely decline with a short statement about your specialization (e.g., "I assist with questions on gender issues in India and cannot address that topic").

Output Requirements:
- Provide only the direct answer or brief declination
- Keep responses focused and concise
- Start your answer immediately
- Do not repeat the user's question
- Do not use prefixes like "ANSWER:" or "GenderWise:"
- Do not explain your methodology
- Do not reference this prompt

Conversation History:
{chat_history}

Context:
{context}

User Question: {question}

Response:"""


FALLBACK_PROMPT_TEMPLATE = """You are GenderWise, an AI assistant focused on gender issues in India.

No relevant documents were retrieved for the current query. Provide a direct, concise answer to the user's question using your general knowledge.

Instructions:
1. Analyze the user's question and conversation history for context
2. Response strategy:
   - For gender issues in India: Provide a brief, factual answer based on your knowledge
   - For questions about yourself: Respond briefly (e.g., "I'm GenderWise, an AI assistant for gender issues in India")
   - For off-topic questions: Politely decline briefly (e.g., "I focus on gender issues in India and cannot help with that")

3. Response requirements:
   - Give only the direct answer
   - Keep it brief and summarized
   - Start your answer immediately
   - Do not repeat the user's question
   - Do not ask new questions
   - Do not use prefixes like "ANSWER:" or "GenderWise:"
   - Do not reference this prompt or conversation history in your response

Conversation History:
{chat_history}

User Question: {question}

Answer:"""

class ModelManager:
    """Manages the LlamaCpp model loading, unloading, and access."""
    _instance = None
    _llm_instance: Optional[LlamaCpp] = None # Store the LlamaCpp instance
    _model_loaded = False
    _load_lock = asyncio.Lock()
    _last_used = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        logger.info("ModelManager initialized.")
        if PRELOAD_MODEL:
             logger.info("PRELOAD_MODEL is true, initiating model load sequence at startup.")
             # Create a task to avoid blocking __init__
             asyncio.create_task(self.load_model())


    async def load_model(self) -> Optional[LlamaCpp]:
        """Loads the LlamaCpp model instance. Thread-safe."""
        async with self._load_lock:
            if self._model_loaded:
                self._last_used = time.time()
                logger.debug("Model already loaded.")
                return self._llm_instance

            logger.info(f"Attempting to load LlamaCpp model from: {MODEL_PATH}...")
            logger.info(f"Using LlamaCpp parameters: n_gpu_layers={N_GPU_LAYERS}, n_batch={N_BATCH}, n_ctx={N_CTX}")

            # Check GPU availability for logging purposes
            gpu_available = torch.cuda.is_available()
            if N_GPU_LAYERS > 0 and not gpu_available:
                logger.warning(f"N_GPU_LAYERS set to {N_GPU_LAYERS}, but CUDA is not available. Will attempt CPU loading.")
            elif N_GPU_LAYERS > 0 and gpu_available:
                 logger.info(f"Attempting to offload {N_GPU_LAYERS} layers to GPU.")


            try:
                # Instantiate LlamaCpp
                stop_sequences = ["\nUSER:", "USER:", "\nUser Question:", "User Question:", "\nASSISTANT:", "\nANSWER:"]
                self._llm_instance = LlamaCpp(
                    model_path=MODEL_PATH,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_NEW_TOKENS,
                    top_p=TOP_P,
                    n_gpu_layers=N_GPU_LAYERS,
                    n_batch=N_BATCH,
                    n_ctx=N_CTX,
                    # repetition_penalty=REPETITION_PENALTY, # Verify exact parameter name if needed
                    f16_kv=True if N_GPU_LAYERS > 0 else False, # Recommended if using GPU layers
                    verbose=os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true",
                    streaming=True,
                    stop=stop_sequences
                )

                self._model_loaded = True
                self._last_used = time.time()
                logger.info(f"LlamaCpp model loaded successfully from {MODEL_PATH}")
                return self._llm_instance

            except Exception as e:
                logger.error(f"Error loading LlamaCpp model: {e}", exc_info=True)
                self._llm_instance = None
                self._model_loaded = False
                return None # Return None to indicate failure

    async def unload_model(self) -> None:
        """Unloads the model to free memory. Thread-safe."""
        async with self._load_lock:
            if not self._model_loaded:
                logger.info("Model already unloaded.")
                return

            logger.info("Unloading LlamaCpp model...")
            try:
                # Delete the LlamaCpp instance
                del self._llm_instance
                self._llm_instance = None

                # Force garbage collection and clear CUDA cache if GPU was used
                gc.collect()
                if N_GPU_LAYERS > 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self._model_loaded = False
                logger.info("LlamaCpp model unloaded successfully.")
            except Exception as e:
                logger.error(f"Error unloading LlamaCpp model: {e}", exc_info=True)
                # Ensure state reflects reality even if cleanup fails partially
                self._model_loaded = False

    async def get_llm_instance(self) -> Optional[LlamaCpp]:
        """Gets the LlamaCpp instance, loading the model if necessary."""
        if not self._model_loaded:
            logger.info("Model not loaded, attempting to load...")
            return await self.load_model() # load_model handles locking
        else:
             self._last_used = time.time() # Update last used time on access
             return self._llm_instance

    async def check_idle(self) -> None:
        """Checks if the model has been idle and unloads it if timeout exceeded."""
        if not self._model_loaded or MODEL_IDLE_TIMEOUT <= 0:
            return

        idle_time = time.time() - self._last_used
        logger.debug(f"Model idle check: Idle for {idle_time:.0f}s / {MODEL_IDLE_TIMEOUT}s")
        if idle_time > MODEL_IDLE_TIMEOUT:
            logger.info(f"Model idle for {idle_time:.0f} seconds (>{MODEL_IDLE_TIMEOUT}s), unloading...")
            await self.unload_model()

    def is_model_loaded(self) -> bool:
        """Checks if the model is currently loaded."""
        return self._model_loaded


class RAGManager:
    """Manages the RAG pipeline and fallback using ModelManager and a retriever."""

    def __init__(self, retriever):
        if retriever is None:
            raise ValueError("RAGManager requires a valid retriever instance.")
        self.retriever = retriever
        self.model_manager = ModelManager()
        # Store both templates
        self.rag_prompt_template = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["chat_history", "context", "question"]
        )
        self.fallback_prompt_template = PromptTemplate(
            template=FALLBACK_PROMPT_TEMPLATE,
            input_variables=["chat_history", "question"]
        )

    async def _get_llm(self):
        """Helper to get the LLM instance, ensures it's loaded."""
        llm_instance = await self.model_manager.get_llm_instance()
        if llm_instance is None:
            logger.error("LLM instance is not available.")
            raise RuntimeError("LLM model failed to load or is unavailable.")
        return llm_instance

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Formats retrieved documents into a single string for the prompt."""
        if not docs:
            return "No relevant information found in documents."
        # Simple concatenation, adjust formatting as needed
        return "\n\n".join([f"Source Chunk ID: {doc.metadata.get('content_hash', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])

    async def stream_query_with_history(
        self, question: str, chat_history: str
    ) -> tuple[AsyncIterator[str], List[Document]]:
        """
        Processes a query using RAG with history, falling back to direct LLM call if needed.

        Args:
            question: The current user question.
            chat_history: Formatted string of recent conversation history.

        Returns:
            A tuple containing:
            - An async iterator yielding response chunks (strings).
            - A list of source documents used (empty if fallback occurred or no docs found).
        """
        llm = await self._get_llm()
        sources = []
        final_prompt = ""

        try:
            # 1. Retrieve relevant documents
            logger.info("Retrieving documents for RAG...")
            # Run blocking retrieval in a thread
            retrieved_docs = await asyncio.to_thread(
                self.retriever.get_relevant_documents, question
            )
            sources = retrieved_docs # Keep track of sources
            logger.info(f"Retrieved {len(sources)} documents.")

            # 2. Decide RAG vs Fallback
            # Fallback if FALLBACK_ENABLED is true AND no documents were found
            use_fallback = FALLBACK_ENABLED and not sources

            if use_fallback:
                logger.info("No relevant documents found or fallback enabled. Using fallback prompt.")
                final_prompt = self.fallback_prompt_template.format(
                    chat_history=chat_history,
                    question=question
                )
                sources = [] # Ensure sources are empty for fallback
            else:
                logger.info("Relevant documents found. Using RAG prompt.")
                context = self._format_docs(sources)
                final_prompt = self.rag_prompt_template.format(
                    chat_history=chat_history,
                    context=context,
                    question=question
                )

            # 3. Stream response from LLM
            logger.debug(f"Streaming LLM with final prompt:\n{final_prompt[:500]}...") # Log start of prompt
            stream_iterator = llm.astream(final_prompt)

            # We need to yield strings, not complex dicts like from QA chains
            async def text_chunk_generator(llm_stream: AsyncIterator[Any]) -> AsyncIterator[str]:
                async for chunk in llm_stream:
                    # LlamaCpp through Langchain typically yields strings directly in astream
                    if isinstance(chunk, str):
                        yield chunk
                    elif hasattr(chunk, 'content') and isinstance(chunk.content, str): # Handle AIMessageChunk etc.
                        yield chunk.content
                    # Add other checks if the LLM stream yields different structures

            return text_chunk_generator(stream_iterator), sources # Return iterator and sources

        except Exception as e:
            logger.error(f"Error during RAG/Fallback stream execution: {e}", exc_info=True)
            # Return an empty iterator and sources on error
            async def empty_gen():
                if False: yield "" # pragma: no cover
            return empty_gen(), []


class MemoryMonitor:
    """Monitors system memory usage."""
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Gets current RAM, Swap, and GPU (if available) memory usage."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            usage = {
                "ram": {
                    "total_gb": round(memory.total / (1024 ** 3), 2),
                    "available_gb": round(memory.available / (1024 ** 3), 2),
                    "used_percent": memory.percent
                },
                "swap": {
                    "total_gb": round(swap.total / (1024 ** 3), 2),
                    "used_gb": round(swap.used / (1024 ** 3), 2),
                    "used_percent": swap.percent
                },
                "gpu": {"available": False},
                "timestamp": time.time()
            }
            if N_GPU_LAYERS > 0 and torch.cuda.is_available():
                usage["gpu"]["available"] = True
                usage["gpu"]["total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                # Could add placeholders for allocated/reserved if needed, but might be misleading.
            return usage
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}

class FallbackHandler:
    """Handles fallback responses and logging."""
    # Simplified to English only
    FALLBACK_RESPONSES = [
        "I'm sorry, I encountered an issue while processing your request. Could you please try again?",
        "I'm having trouble finding that information right now. Please try rephrasing your question.",
        "My apologies, I couldn't generate a response for that query.",
    ]
    LOG_FILE = "fallback_logs.jsonl"

    @staticmethod
    def get_fallback_response() -> str: # Removed language parameter
        """Gets a random fallback response."""
        return np.random.choice(FallbackHandler.FALLBACK_RESPONSES)

    @staticmethod
    async def log_fallback(user_query: str, error: Exception, conversation_id: Optional[str] = None):
        """Asynchronously logs fallback details to a file."""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": user_query,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "conversation_id": str(conversation_id) if conversation_id else None,
                "model_loaded": ModelManager().is_model_loaded(), # Use instance method
                "memory": MemoryMonitor.get_memory_usage() # Log memory state at time of error
            }
            async with aiofiles.open(FallbackHandler.LOG_FILE, mode="a", encoding="utf-8") as f:
                await f.write(json.dumps(log_entry) + "\n")
        except Exception as log_e:
            logger.error(f"Critical Error: Failed to log fallback event: {log_e}")

def create_error_response(message: str, code: str = "INTERNAL_ERROR", status_code: int = 500) -> Dict[str, Any]:
    """Creates a standardized JSON error response."""
    return {
        "detail": {
            "code": code,
            "message": message,
        }
    }

def generate_uuid() -> uuid.UUID:
    """Generates a UUID object."""
    return uuid.uuid4()

def get_model_manager():
    return ModelManager()