import os
import json
import logging
from datetime import datetime
from typing import List, Dict
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Set environment variables
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Load API key
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self,
        persist_directory: str = "./chroma_db",
        memory_file: str = "long_term_notes.json",
        chat_history_file: str = "chat_history.json"):

        self.embeddings = OpenAIEmbeddings(api_key=API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_store = None
        self.persist_directory = persist_directory

        self.memory_file = memory_file
        self.chat_history_file = chat_history_file

        self.long_term_notes = []
        self.session_notes = []
        self.chat_history = []

        self._load_long_term_memory()
        self._load_chat_history()

    def initialize_vector_store(self):
        self.vector_store = Chroma(
            collection_name="memory",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.info("✅ Vector store initialized")

    def add_pdf_to_memory(self, pdf_content: str, session_id: str):
        """Processes and stores PDF content and embeddings manually without using add_texts"""

        timestamp = datetime.now().isoformat()
        session_entry = {
            "content": pdf_content,
            "timestamp": timestamp,
            "session_id": session_id
        }

        self.session_notes.append(session_entry)
        self.long_term_notes.append(session_entry)

        try:
            chunks = self.text_splitter.split_text(pdf_content)
        except Exception as e:
            logger.error(f"❌ Error while splitting text: {e}")
            return

        if not chunks:
            logger.warning("⚠️ No chunks generated from PDF content.")
            return

        logger.info(f"🧩 Total chunks generated: {len(chunks)}")

        if self.vector_store is None:
            try:
                self.initialize_vector_store()
            except Exception as e:
                logger.error(f"❌ Failed to initialize vector store: {e}")
                return

        try:
            # Manually embed the chunks
            logger.info("🔍 Generating embeddings for chunks...")
            embeddings = self.embeddings.embed_documents(chunks)
            logger.info("✅ Embeddings generated")

            # Create metadata for each chunk
            metadatas = [{"session_id": session_id, "chunk_id": i} for i in range(len(chunks))]
            ids = [str(session_id) + "_" + str(i) for i in range(len(chunks))]

            # Add embeddings to the vector store
            logger.info("💾 Adding embeddings to vector store...")
            self.vector_store._collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("✅ Chunks and embeddings added to vector store")
        except Exception as e:
            logger.exception("❌ Error while manually adding embeddings to vector store")
            return

        try:
            self.save_memory()
        except Exception as e:
            logger.warning(f"⚠️ Memory saved partially due to error: {e}")


    def retrieve_relevant_context(self, query: str, k: int = 5) -> str:
        if self.vector_store is None:
            return ""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_notes, f, indent=2, ensure_ascii=False)

            os.makedirs(os.path.dirname(self.chat_history_file) or ".", exist_ok=True)
            with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)

            logger.info("💾 Memory saved")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving memory: {e}")
            return False

    def _load_long_term_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.long_term_notes = json.load(f)
        except Exception as e:
            logger.error(f"Error loading long-term memory: {e}")
            self.long_term_notes = []

    def _load_chat_history(self):
        try:
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            self.chat_history = []

    def add_to_chat_history(self, message: Dict):
        self.chat_history.append({
            **message,
            "timestamp": datetime.now().isoformat()
        })
        self.save_memory()

    def get_memory_stats(self) -> Dict:
        return {
            "long_term_notes": len(self.long_term_notes),
            "session_notes": len(self.session_notes),
            "chat_history": len(self.chat_history),
            "vector_store_initialized": self.vector_store is not None
        }
    def get_optimized_history(self, max_tokens: int = 2000) -> List[Dict]:
        """Get optimized chat history within token limit"""
        encoding = tiktoken.get_encoding("cl100k_base")
        optimized_history = []
        current_tokens = 0
        
        # Start from most recent messages
        for message in reversed(self.chat_history):
            message_tokens = len(encoding.encode(str(message)))
            if current_tokens + message_tokens > max_tokens:
                break
            optimized_history.insert(0, message)
            current_tokens += message_tokens
        
        return optimized_history
    
    def clear_session_memory(self):
        """Clear session-specific memory while keeping long-term memory and chat history"""
        self.session_notes = []
        logger.info("Session memory cleared")
    
    def clear_all_chat_history(self):
        """Clear all chat history and save"""
        self.chat_history = []
        self.save_memory()
        logger.info("All chat history cleared and saved")
    
    def backup_memory(self, backup_path: str = None):
        """Create a backup of the memory file"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.memory_file}.backup_{timestamp}"
        
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as src:
                    with open(backup_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                logger.info(f"Memory backup created at {backup_path}")
                return backup_path
            else:
                logger.warning("No memory file exists to backup")
                return None
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None