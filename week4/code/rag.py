from langchain_openai import OpenAIEmbeddings
import chromadb
import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
def load_and_chunk_doc(file_path):
    """load and chunk PDF file. """   
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    if not docs:
        logging.warning("Failed to load document.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)
    return splits

def generate_embeddings_and_vectordb(splits):
    """Generate embeddings from document chunks and vector database with Chroma"""
    try:
        embedding = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                embedding_ctx_length=3000,
        )
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
        return vectorstore
    
    except Exception as e:
        logging.warning(f"Failed to create vector database: {str(e)}")
        return None
    
def retrieve_chunks(query, vectorstore, k=2) -> str:
    """Retrieve top-k relevant document chunks"""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)
        notes = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return notes
    
    except Exception as e:
        logging.error(f"Failed to retrieve relevant documents: {str(e)}")
        return "No relevant information found in your notes."
    
def prep_ragdb():
    """Generate vectordb for all PDFs in NOTES directory."""
    # File path
    dir_path = r"week4\notes"

    logging.info("  >> Breaking your notes into smaller, manageable pieces...")
    if os.path.exists(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            chunked_docs= load_and_chunk_doc(file_path)
    
    logging.info("  >> Creating embeddings and vector database to enable smart search from your notes...")
    vectorstore = generate_embeddings_and_vectordb(chunked_docs)
    return vectorstore
         
