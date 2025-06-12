from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time

# Load Environment Variables
load_dotenv()

# Validate OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")    

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Load Markdown Notes
def load_doc(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                notes = f.read()
            return notes
        except Exception as e:
            print(f"Failed to read the file: {str(e)}")
            return None
    else:
        print(f"File path - '{file_path}' doesn't exist.")
        return None

# Initiate LLM   
def load_llm(temperature = 0.5, model_name = "gpt-3.5-turbo"):
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

    # File path
    file_path = r"C:\Users\debli\OneDrive\Desktop\AI_Agents\my_notes.md"

    # Step1 : Load the documents
    notes = load_doc(file_path)
    if not notes:
        return

    #Step2 : Initiate LLM
    llm = load_llm(temperature=0.4, model_name="gpt-3.5-turbo")

    while True:

        # Step3 : Take user question as input
        question = input("\nEnter your question (or type 'exit' to quit):")
        if question.lower() == "exit":
            break

        #Step4 : Create prompt
        prompt = create_prompt(notes, question)
    
        # Step5: Generate anwser from LLM
        start = time.time()
        response = llm.invoke(input = prompt)
        response_time = time.time() - start

        # Step6 : Extract content from reponse
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response).strip()

        print(f"\nStudy Buddy: {answer}")
        print(f"\nLLM response time : {response_time:.2f} seconds")

if __name__ == "__main__" :
    main()
