from datetime import datetime, date
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
from langchain.vectorstores import Chroma
import logging
import tiktoken
import json
import os
import openai
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages long-term, short-term, and vector-based memory"""
    
    def __init__(self, persist_directory: str = "./chroma_db", memory_file: str = "long_term_notes.json", chat_history_file: str = "chat_history.json"):
        self.embeddings = OpenAIEmbeddings(api_key=API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self.persist_directory = persist_directory
        self.memory_file = memory_file
        self.chat_history_file = chat_history_file
        self.long_term_notes = []
        self.session_notes = []
        self.chat_history = []
        
        # Load existing data on initialization
        self._load_long_term_memory()
        self._load_chat_history()
        
    def initialize_vector_store(self):
        """Initialize vector database for current session"""
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def _load_long_term_memory(self):
        """Load long-term memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.long_term_notes = json.load(f)
                logger.info(f"Loaded {len(self.long_term_notes)} long-term memory entries")
            else:
                logger.info("No existing long-term memory file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading long-term memory: {e}")
            self.long_term_notes = []
    
    def _load_chat_history(self):
        """Load chat history from file"""
        try:
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                    print(self.chat_history_file)
                logger.info(f"Loaded {len(self.chat_history)} chat history entries")
            else:
                logger.info("No existing chat history file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            self.chat_history = []
    
    def save_memory(self):
        """Save long-term memory, chat history, and persist vector database"""
        try:
            # Save long-term memory to JSON file
            os.makedirs(os.path.dirname(self.memory_file) if os.path.dirname(self.memory_file) else '.', exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_notes, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.long_term_notes)} long-term memory entries to {self.memory_file}")
            
            # Save chat history to JSON file
            os.makedirs(os.path.dirname(self.chat_history_file) if os.path.dirname(self.chat_history_file) else '.', exist_ok=True)
            with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.chat_history)} chat history entries to {self.chat_history_file}")

            # Persist vector database
            if self.vector_store is not None:
                self.vector_store.persist()
                logger.info("Vector database persisted successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def add_pdf_to_memory(self, pdf_content: str, session_id: str):
        """Process and store PDF content in memory systems"""
        # Add to session notes
        self.session_notes.append({
            "content": pdf_content,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        })
        
        # Add to long-term notes (persistent)
        self.long_term_notes.append({
            "content": pdf_content,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        })
        
        # Process for vector database
        chunks = self.text_splitter.split_text(pdf_content)
        if self.vector_store is None:
            self.initialize_vector_store()
        
        # Add chunks to vector database with metadata
        metadatas = [{"session_id": session_id, "chunk_id": i} for i in range(len(chunks))]
        self.vector_store.add_texts(chunks, metadatas=metadatas)
        
        logger.info(f"Added {len(chunks)} chunks to vector database")
        
        # Auto-save after adding new content
        self.save_memory()
    
    def retrieve_relevant_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from vector database"""
        if self.vector_store is None:
            return ""
        
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    
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
    
    def add_to_chat_history(self, message: Dict):
        """Add a message to chat history and save"""
        self.chat_history.append({
            **message,
            "timestamp": datetime.now().isoformat()
        })
        # Auto-save chat history
        self.save_memory()
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories"""
        return {
            "long_term_notes_count": len(self.long_term_notes),
            "session_notes_count": len(self.session_notes),
            "chat_history_count": len(self.chat_history),
            "vector_store_initialized": self.vector_store is not None,
            "memory_file_exists": os.path.exists(self.memory_file),
            "chat_history_file_exists": os.path.exists(self.chat_history_file),
            "persist_directory_exists": os.path.exists(self.persist_directory)
        }
    
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


