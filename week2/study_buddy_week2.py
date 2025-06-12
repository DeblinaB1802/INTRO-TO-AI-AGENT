from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
import chromadb
import os
import time
import PyPDF2
from pathlib import Path

# Load Environment Variables
load_dotenv()

# Validate OpenAI API key
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")    

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Load pdf files and convert them in small document chunks
def load_and_chunk_doc(file_path):
    if os.path.exists(file_path):
        file_path = Path(file_path)       
        if file_path.suffix == ".pdf":
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    chunked_docs = []
                    metadatas = []
                    ids = []
                    for page_num, page in enumerate(reader.pages, start=1):
                        text = page.extract_text() or ""
                        chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk) > 10]
                        for chunk in chunks:
                            doc_id = str(uuid.uuid4())
                            chunked_docs.append(chunk)
                            metadatas.append({
                                    "source" : file_path.name,
                                    "page_number" : page_num,
                                    "chunk_size" : len(chunk),
                                })
                            ids.append(doc_id)
                return chunked_docs, metadatas, ids
            
            except Exception as e:
                print(f"Failed to read the file: {str(e)}")
                return None, None, None
    else:
        print(f"File path - '{file_path}' doesn't exist.")
        return None, None, None

# Generate embeddings
def generate_embeddings(chunked_docs):
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(chunked_docs, convert_to_numpy=True)
        return embeddings
    
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return None

# Create vector database with Chroma    
def vector_db(chunked_docs, metadatas, ids, embeddings):
    chroma_client = chromadb.Client()
    vectorstore = chroma_client.create_collection(name="pdf_chunks")
    vectorstore.add(
        documents=chunked_docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    return vectorstore

# Retrieve top-k relevant document chunks
def retrieve_chunks(question, vectorstore, k=2):
    retrieved_docs = vectorstore.query(
        query_texts = [question],
        n_results = k,
        include = ['documents', 'metadatas', 'distances']
    )
    return retrieved_docs

# Initiate LLM   
def load_llm(temperature = 0.2, model_name = "gpt-3.5-turbo"):
    try:
        llm = ChatOpenAI(temperature=temperature, model=model_name)
        print("Model successfully initialized...")
        return llm
    except Exception as e:
        print(f"Failed to initiate model: {str(e)}")
        return None
    
# Create Prompt Template
def create_prompt(notes, question):
    TEMPLATE = """
            You are an AI Study Assistant. answer the following question ONLY based on the notes provided to you. 
            If you can't find the answer in the notes, state that "I can't find the answer in the provided notes"
            
            Notes : {notes}

            Question : {question}
            Answer:
        """
    prompt = PromptTemplate.from_template(TEMPLATE)
    return prompt.format(notes=notes, question=question)

# Main Function
def main():
    print("Welcome to Study Buddy — your partner in learning. Let’s explore knowledge together!")
    print("\n\nJust a moment! Your Study Buddy is getting everything ready.")

    # File path
    dir_path = r"c:\Users\debli\OneDrive\Desktop\AI_Agents\notes"

    # Step1 : Load and chunk the documents
    print("  >> Breaking your notes into smaller, manageable pieces...")
    all_chunked_docs, all_metadatas, all_ids = [], [], []
    if os.path.exists(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            chunked_docs, metadatas, ids = load_and_chunk_doc(file_path)
            all_chunked_docs.extend(chunked_docs)
            all_metadatas.extend(metadatas)
            all_ids.extend(ids)
    else:
        print(f"Direcory path - {dir_path} doesn't exist.")
        return
    
    if not all_chunked_docs:
        print("No documents were chunked.")
        return

    # Step2 : Generate vector embeddings
    print("  >> Creating embeddings to understand your notes better...")
    embeddings = generate_embeddings(all_chunked_docs)

    # Step3 : Create vector database
    print("  >> Setting up a smart search system with your notes...")
    vectorstore = vector_db(all_chunked_docs, all_metadatas, all_ids, embeddings)

    # Step4 : Initiate LLM
    llm = load_llm(temperature=0.2, model_name="gpt-3.5-turbo")
    if not llm:
        return

    print("\n\nAll set! Your Study Buddy is now ready to assist you.")
    
    while True:
        # Step5 : Take user question as input
        question = input("\nEnter your question (or type 'exit' to quit):")
        if question.lower() == "exit":
            print("Good Bye!!!")
            break

        # Step6 : Retrieve relevant document chunks
        retrieve_docs = retrieve_chunks(question, vectorstore, k=2)
        if not retrieve_docs['documents'] or not retrieve_docs['documents'][0]:
            print("No relevant chunks found.")
            continue

        notes = [retrieve_docs['documents'][0][i] for i in range(len(retrieve_docs['documents'][0]))]
        notes = '\n'.join(notes)

        # Step7 : Create prompt
        prompt = create_prompt(notes, question)
    
        # Step8: Generate anwser from LLM
        start = time.time()
        response = llm.invoke(prompt)
        response_time = time.time() - start

        # Step9 : Extract content from reponse
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response).strip()

        # Step10 : Show the response and  top-k retrieved documents
        print(f"\nStudy Buddy: {answer}")
        print(f"\nLLM response time : {response_time:.2f} seconds")
        print(f"\n____RETRIEVED DOCUMENT DETAILS____")
        for i in range(len(retrieve_docs['documents'][0])):
            similarity = 1 - retrieve_docs['distances'][0][i]
            print(f"\n     Retrieved Document {i+1}:")
            print(f"Source : {retrieve_docs['metadatas'][0][i]['source']}")
            print(f"Page : {retrieve_docs['metadatas'][0][i]['page_number']}")
            print(f"Document chunk size : {retrieve_docs['metadatas'][0][i]['chunk_size']}")
            print(f"Cosine similarity : {similarity:.4f}")
            print(f"Content : {retrieve_docs['documents'][0][i][:200]}")

if __name__ == "__main__" :
    main()
