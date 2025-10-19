import os
import pandas as pd
import ast
import traceback # Import the traceback module
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Import LangChain & Environment components ---
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# --- 1. App & CORS Setup ---

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost",
    "https://piyush.mankiratsingh.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Pydantic Models (Data Validation) ---
class UserQuery(BaseModel):
    query: str

# --- 3. LangChain RAG Setup ---

try:
    # 1. Embedding Model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. LLM (Using Groq for fast generation) - FINAL STABLE MODEL
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct", # Switched to a stable, supported Mixtral model
        temperature=0.7
    )

    # 3. Pinecone Vector Store
    index_name = os.getenv("PINECONE_INDEX_NAME")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key='original_text' # Explicitly tell LangChain which metadata field to use for content
    )
    
    # 4. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    
    print("LangChain components initialized successfully.")

except Exception as e:
    print(f"Error initializing LangChain components: {e}")
    retriever = None
    llm = None

# B. Define Prompt Template (used for generation)
template = """
You are a creative and persuasive furniture salesperson.
A customer is asking for a product. You have found a potential match.
Your job is to write a short, creative, and inspiring product description
(2-3 sentences) based on the customer's query and the product's details.

Customer Query: {query}

Product Details:
{context}

Creative Description:
"""

prompt = PromptTemplate.from_template(template)

# C. Define the Output Parser
output_parser = StrOutputParser()


# --- 4. API Endpoints ---

@app.post("/recommend")
async def get_recommendation(user_query: UserQuery):
    """
    Endpoint to get product recommendations.
    """
    if retriever is None or llm is None:
        return {"error": "AI components not initialized."}

    print(f"Received query: {user_query.query}")
    
    try:
        # Step 1: Retrieve relevant documents from Pinecone
        retrieved_docs = retriever.invoke(user_query.query)
        
        recommendations = []
        
        # Step 2: For each retrieved product, generate a new description
        for doc in retrieved_docs:
            product_metadata = doc.metadata
            product_context = product_metadata.get('original_text', '')
            
            response = "" # Initialize response
            try:
                # --- ROBUST MANUAL CHAIN EXECUTION ---
                # 3a. Format the prompt manually with product-specific context
                formatted_prompt = prompt.format(
                    query=user_query.query, 
                    context=product_context
                )
                
                # 3b. Invoke the LLM directly with the formatted prompt string
                llm_result = llm.invoke(formatted_prompt)
                
                # 3c. **Crucial Check**: Inspect the result before parsing
                if hasattr(llm_result, 'content'):
                    # It's a valid message object, so we can safely get the content.
                    response = llm_result.content
                else:
                    # It's not a valid object, likely an error dict from the API.
                    print(f"  -> LLM call failed or returned unexpected type. API returned: {llm_result}")
                    raise ValueError("LLM did not return a valid message object.")

            except Exception as gen_e:
                print(f"Could not generate description for product: {product_metadata.get('title', 'Unknown')}.")
                print(f"  -> Generation Error: {gen_e}") # Log the specific error
                response = "A creative description for this product is coming soon!" # Fallback description

            # Step 4: Add the generated (or fallback) description to our product data
            product_metadata['generated_description'] = response
            recommendations.append(product_metadata)
            
        return {"recommendations": recommendations}

    except Exception as e:
        # Enhanced error logging for broader issues (e.g., retrieval failure)
        print(f"Error during retrieval step: {e}")
        print("--- FULL TRACEBACK ---")
        traceback.print_exc() # Print the detailed traceback to the console
        print("--- END TRACEBACK ---")
        return {"error": f"Failed to retrieve products. Check backend logs for details."}


@app.get("/analytics")
def get_analytics():
    """
    Endpoint to serve aggregated data for the analytics dashboard.
    """
    print("Analytics endpoint pinged")
    return load_analytics_data()


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Product Recommendation API!"}

# Helper function for analytics
def load_analytics_data():
    """
    Loads and prepares the analytics data from our CSV.
    """
    file_path = '../intern_data_ikarus.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return {"error": "Dataset not found. Make sure 'intern_data_ikarus.csv' is in the root."}

    def get_primary_category(categories_str):
        try:
            categories_list = ast.literal_eval(categories_str)
            if isinstance(categories_list, list) and len(categories_list) > 0:
                return categories_list[0]
        except:
            pass
        return 'Unknown'

    df['price'] = df['price'].replace({r'$': '', ',': ''}, regex=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['primary_category'] = df['categories'].apply(get_primary_category)

    category_counts = df['primary_category'].value_counts().head(10).to_dict()
    brand_counts = df['brand'].value_counts().head(10).to_dict()
    material_counts = df['material'].fillna('Unknown').value_counts().head(10).to_dict()
    
    analytics_json = {
        "category_counts": [{"name": k, "value": v} for k, v in category_counts.items()],
        "brand_counts": [{"name": k, "value": v} for k, v in brand_counts.items()],
        "material_counts": [{"name": k, "value": v} for k, v in material_counts.items()],
    }
    
    return analytics_json

