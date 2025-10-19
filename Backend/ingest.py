import pandas as pd
import ast
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # For a nice progress bar

# --- 1. Pinecone Configuration ---
# !! ACTION REQUIRED: Fill these in with your Pinecone details !!
PINECONE_API_KEY = "pcsk_4hkz8N_BdjwnnbdqRj5pQKyTCPqRGmqodFhiXW29NnwHUZDKb8sMd95QUqNV9UhQcefaCL"
PINECONE_INDEX_NAME = "3d-ikraus"
# e.g., "gcp-starter", "us-east-1", etc. Find this in your Pinecone console.
PINECONE_ENVIRONMENT = "us-east-1" 
MODEL_DIMENSIONS = 384 # This is fixed for 'all-MiniLM-L6-v2'
BATCH_SIZE = 100 # Process 100 items at a time

# --- 2. Data Loading and Cleaning Function ---

def get_first_image_url(images_str):
    """Parses the string list of images and returns the first URL."""
    try:
        images_list = ast.literal_eval(images_str)
        if isinstance(images_list, list) and len(images_list) > 0:
            return images_list[0].strip() # Return the first image URL
    except:
        pass
    return None # Return None if parsing fails or list is empty

def get_primary_category(categories_str):
    """Parses the string list of categories and returns the first one."""
    try:
        categories_list = ast.literal_eval(categories_str)
        if isinstance(categories_list, list) and len(categories_list) > 0:
            return categories_list[0]
    except:
        pass
    return 'Unknown'

def load_and_clean_data(file_path=r'C:\Users\piyus\OneDrive\Desktop\AI_Ikarus3D\intern_data_ikarus.csv'):
    """Loads and prepares the dataset for ingestion."""
    df = pd.read_csv(file_path)
    
    # 1. Handle NaNs by filling with placeholder text
    # This ensures all products have some text to be embedded.
    df['description'] = df['description'].fillna('No description available')
    df['material'] = df['material'].fillna('Unknown')
    df['color'] = df['color'].fillna('Unknown')
    
    # 2. Clean Price
    df['price_str'] = df['price'].astype(str) # Keep string for display
    df['price'] = df['price'].replace({'\$': '', ',': ''}, regex=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price'] = df['price'].fillna(0.0) # Use 0.0 for missing prices
    
    # 3. Parse Categories and Images
    df['primary_category'] = df['categories'].apply(get_primary_category)
    df['image_url'] = df['images'].apply(get_first_image_url)
    
    # 4. Create the consolidated text for embedding (Our NLP Strategy)
    # We combine the most important fields into one string for semantic search.
    df['embedding_text'] = (
        "Title: " + df['title'] +
        ". Brand: " + df['brand'] +
        ". Category: " + df['primary_category'] +
        ". Material: " + df['material'] +
        ". Color: " + df['color'] +
        ". Description: " + df['description']
    )
    
    # 5. Drop rows where we couldn't get a unique ID or embedding text
    df = df.dropna(subset=['uniq_id', 'embedding_text'])
    
    return df

# --- 3. Pinecone Ingestion ---

def ingest_data():
    """Main function to load data, generate embeddings, and upsert to Pinecone."""
    
    print("Loading and cleaning data...")
    df = load_and_clean_data()
    
    print(f"Loaded {len(df)} products to ingest.")
    
    print("Initializing Pinecone connection...")
    if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_PINECONE_API_KEY":
        print("Error: PINECONE_API_KEY is not set. Please edit the script.")
        return

    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists and create it if not
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating it...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=MODEL_DIMENSIONS,
                metric='dotproduct',
                spec=ServerlessSpec(
                    cloud='aws', # 'aws' or 'gcp'
                    region=PINECONE_ENVIRONMENT # e.g., 'us-east-1'
                )
            )
            print(f"Index created successfully. Please wait a moment for it to initialize.")
        except Exception as e:
            print(f"Error creating index: {e}")
            return
    else:
        print(f"Found existing index '{PINECONE_INDEX_NAME}'.")

    index = pc.Index(PINECONE_INDEX_NAME)
    
    print("Initializing Sentence Transformer model ('all-MiniLM-L6-v2')...")
    # This model is fast and has 384 dimensions
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings and upserting to Pinecone in batches...")
    
    # Process in batches
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        
        # 1. Get embedding text
        texts_to_embed = batch_df['embedding_text'].tolist()
        
        # 2. Generate embeddings
        embeddings = model.encode(texts_to_embed, show_progress_bar=False).tolist()
        
        # 3. Prepare metadata
        # This is the data we'll get back from Pinecone after a search
        metadata_batch = []
        for _, row in batch_df.iterrows():
            metadata_batch.append({
                'title': row['title'],
                'brand': row['brand'],
                'price': float(row['price']),
                'category': row['primary_category'],
                'image_url': row['image_url'],
                'original_text': row['embedding_text'] # Store for context
            })
        
        # 4. Prepare vectors for upsert
        # The 'id' must be a string
        ids_batch = batch_df['uniq_id'].tolist()
        vectors_batch = list(zip(ids_batch, embeddings, metadata_batch))
        
        # 5. Upsert to Pinecone
        try:
            index.upsert(vectors=vectors_batch)
        except Exception as e:
            print(f"Error upserting batch {i}: {e}")
            
    print("\n--- Data Ingestion Complete! ---")
    print(index.describe_index_stats())

if __name__ == "__main__":
    ingest_data()