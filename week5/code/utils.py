import PyPDF2
import os
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Additional helper functions for deployment

class PDFProcessor:
    """Utility class for processing PDF files"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text content from PDF file"""

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            return ""
        
class ErrorHandler:
    """Centralized error handling and recovery"""
    
    @staticmethod
    def handle_api_error(error: Exception, tool_name: str) -> Tuple[str, float]:
        """Handle API-related errors gracefully"""

        error_message = f"Service temporarily unavailable for {tool_name}"
        logger.error(f"{tool_name} API error: {str(error)}")
        return error_message
    
    @staticmethod
    def handle_processing_error(error: Exception, context: str) -> str:
        """Handle processing errors with user-friendly messages"""

        logger.error(f"Processing error in {context}: {str(error)}")
        return f"I encountered an issue while {context}. Please try rephrasing your request."
    
# Performance monitoring
class PerformanceMonitor:
    """Monitor system performance and usage"""
    
    def __init__(self):
        self.query_count = 0
        self.response_times = []
        self.error_count = 0
        self.start_time = datetime.now()
    
    def log_query(self, response_time: float, error: bool = False):
        """Log query performance metrics"""

        self.query_count += 1
        self.response_times.append(response_time)
        if error:
            self.error_count += 1
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        uptime = datetime.now() - self.start_time
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        
        return {
            "uptime": str(uptime),
            "total_queries": self.query_count,
            "average_response_time": f"{avg_response_time:.2f}s",
            "error_rate": f"{(self.error_count/self.query_count*100):.1f}%" if self.query_count > 0 else "0%",
            "errors": self.error_count
        }