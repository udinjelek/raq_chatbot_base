# rag_chatbot.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Configuration
DOCS_DIR = "company_docs"
VECTOR_STORE_PATH = "vectorstore"

def load_and_process_documents():
    """Load and split documents from the company_docs directory"""
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents)

def setup_vectorstore(docs):
    """Create and save FAISS vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return vectorstore

def load_vectorstore():
    """Load existing FAISS vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

def get_relevant_context(vectorstore, query):
    """Retrieve relevant context from documents"""
    docs = vectorstore.similarity_search(query, k=4)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_response(query, context):
    """Generate response using Gemini model with context"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        chat = model.start_chat()

        prompt = f"""
You are a helpful company assistant. Answer the user's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have information about that in our documents."

Context:
{context}

Question: {query}
Answer:
        """

        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"\nError generating response: {str(e)}")
        return "I encountered an error processing your request. Please try again."



def main():
    # Load/process documents if not already vectorized
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Processing documents...")
        docs = load_and_process_documents()
        vectorstore = setup_vectorstore(docs)
    else:
        print("Loading existing vector database...")
        vectorstore = load_vectorstore()
    
    print("\nCompany Chatbot ready! (type 'exit' to quit)")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == "exit":
            break
        
        print("\nSearching documents...")
        context = get_relevant_context(vectorstore, query)
        
        # Debug: Show context snippets
        # print(f"\nFound {len(context.split('\\n\\n'))} relevant document snippets")
        # print(f"\nFound {len(context.split('\n\n'))} relevant document snippets")
        separator = '\n\n'
        print(f"\nFound {len(context.split(separator))} relevant document snippets")


        
        print("Generating response...")
        response = generate_response(query, context)
        
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()