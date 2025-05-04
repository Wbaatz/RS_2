from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time
from pymongo import MongoClient
from pinecone_text.sparse import BM25Encoder
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
from tqdm.auto import tqdm
import pickle
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Hybrid Image Search API")

# Load environment variables
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (change this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PINECONE_API_KEY = "pcsk_5h3nEa_DiCLbqeEAGLnYKGsuYWyE75iu214MWPfpic3QTxM3D9stev7kBw1ArVgEAETaWd"  # Replace with your Pinecone API key
INDEX_NAME ='hybrid-image-search'
MONGO_URI = "mongodb+srv://mmehdibhojani1:kNFhOK1vE1roDhTS@cluster0.a6tbh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
BM25_MODEL_PATH = "bm25_model.pkl"


# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
# Connect to Pinecone index
def get_pinecone_index():
    if INDEX_NAME not in [index.name for index in pinecone.list_indexes()]:
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=512,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pinecone.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
    return pinecone.Index(INDEX_NAME)

index = get_pinecone_index()

# Initialize MongoDB
client = MongoClient(MONGO_URI)
db = client["test"]
products_collection = db["products"]
results_collection = db["search_results"]

# Initialize SentenceTransformer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)


def fetch_data_from_mongodb():
    try:
        # Retrieve all products
        products = list(products_collection.find({}))
        if not products:
            raise ValueError("No products found in MongoDB")

        # Extract metadata and images
        metadata = []
        image_urls = []
        for product in products:
            # Use description as the text field for embeddings
            metadata.append({
                "productId": product["productId"],
                "description": product["description"],
                "tags": product["tags"],
                "name": product["name"]
            })
            # Assume img is an array; take the first URL
            image_urls.append(product["img"][0] if product["img"] else None)

        # Filter out products with missing image URLs
        valid_data = [(m, url) for m, url in zip(metadata, image_urls) if url]
        if not valid_data:
            raise ValueError("No valid products with image URLs found")

        metadata, image_urls = zip(*valid_data)
        metadata = pd.DataFrame(metadata)
        return metadata, image_urls
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return None, None


def get_bm25_model(metadata):
    try:
        # Check if BM25 model file exists
        if os.path.exists(BM25_MODEL_PATH):
            with open(BM25_MODEL_PATH, 'rb') as f:
                bm25 = pickle.load(f)
                print(f"Loaded BM25 model from {BM25_MODEL_PATH}")
        else:
            # Train new BM25 model
            bm25 = BM25Encoder()
            bm25.fit(metadata['description'])
            # Save the model
            with open(BM25_MODEL_PATH, 'wb') as f:
                pickle.dump(bm25, f)
                print(f"Saved BM25 model to {BM25_MODEL_PATH}")
        return bm25
    except Exception as e:
        raise Exception(f"Error handling BM25 model: {e}")

# Step 2: Train the Recommendation System
def train_recommendation_system(metadata, image_urls):
    try:
        # Initialize BM25Encoder
        bm25 = get_bm25_model(metadata)

        # Create or connect to Pinecone index
        if INDEX_NAME not in pinecone.list_indexes().names():
            pinecone.create_index(
                INDEX_NAME,
                dimension=512,
                metric='dotproduct',
                spec=spec
            )
            while not pinecone.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)

        index = pinecone.Index(INDEX_NAME)

        # Process data in batches
        batch_size = 200
        for i in tqdm(range(0, len(metadata), batch_size), desc="Indexing"):
            i_end = min(i + batch_size, len(metadata))
            meta_batch = metadata.iloc[i:i_end]
            meta_dict = meta_batch.to_dict(orient="records")
            img_urls_batch = image_urls[i:i_end]

            # Download and encode images
            images = []
            for url in img_urls_batch:
                try:
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Error downloading image {url}: {e}")
                    continue

            # Create sparse and dense embeddings
            text_batch = meta_batch['description'].tolist()
            sparse_embeds = bm25.encode_documents(text_batch)
            dense_embeds = model.encode(images).tolist()
            ids = [meta['productId'] for meta in meta_dict]

            # Prepare upserts
            upserts = []
            for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict):
                upserts.append({
                    'id': _id,
                    'sparse_values': sparse,
                    'values': dense,
                    'metadata': {
                        'productId': meta['productId'],
                        'name': meta['name'],
                        'description': meta['description']
                    }
                })

            # Upsert to Pinecone
            index.upsert(upserts)

        print(f"Indexed {len(metadata)} products in Pinecone")
        return bm25, index
    except Exception as e:
        print(f"Error training recommendation system: {e}")
        return None, None








# Step 3: Save the Model
def save_model(bm25, save_path="bm25_model.pkl"):
    try:
        # Save BM25Encoder
        with open(save_path, 'wb') as f:
            pickle.dump(bm25, f)
        print(f"BM25 model saved to {save_path}")

        # Note: SentenceTransformer is pre-trained; no need to save unless fine-tuned
        # Pinecone index persists in the cloud
    except Exception as e:
        print(f"Error saving model: {e}")

# Load BM25 model
def load_bm25_model():
    try:
        with open(BM25_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading BM25 model: {e}")

# bm25 = load_bm25_model()



# Pydantic models for request/response validation
class SearchRequest(BaseModel):
    tags: List[str]
    image_url: str
    top_k: int = 14
    alpha: float = 0.5

class SearchResponse(BaseModel):
    product_ids: List[str]



# Step 4: Query with Prompt and Image URL
def query_recommendation(prompt, image_url, bm25, index, top_k=14, alpha=0.5):
    try:
        # Encode prompt
        sparse = bm25.encode_queries(prompt)
        dense_text = model.encode(prompt).tolist()

        # Download and encode image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        dense_image = model.encode(img).tolist()

        # Combine dense vectors (average text and image)
        dense = [(t + i) / 2 for t, i in zip(dense_text, dense_image)]

        # Scale sparse and dense vectors
        def hybrid_scale(dense, sparse, alpha):
            if not 0 <= alpha <= 1:
                raise ValueError("Alpha must be between 0 and 1")
            hsparse = {
                'indices': sparse['indices'],
                'values': [v * (1 - alpha) for v in sparse['values']]
            }
            hdense = [v * alpha for v in dense]
            return hdense, hsparse

        hdense, hsparse = hybrid_scale(dense, sparse, alpha)

        # Query Pinecone
        result = index.query(
            top_k=top_k,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True
        )

        # Extract product IDs
        product_ids = [r["id"] for r in result["matches"]]

        # Fetch product details from MongoDB
        products = list(products_collection.find({"productId": {"$in": product_ids}}))
        product_details = {p["productId"]: p for p in products}

        # Prepare results
        search_results = []
        for r in result["matches"]:
            product_id = r["id"]
            product = product_details.get(product_id, {})
            search_results.append({
                "productId": product_id,
                "name": product.get("name", ""),
                "description": product.get("description", ""),
                "image_url": product.get("img", [None])[0],
                "score": r["score"],
                "query_prompt": prompt,
                "query_image_url": image_url,
                "timestamp": time.time()
            })

        # Store results in MongoDB
        if search_results:
            results_collection.insert_many(search_results)
            print(f"Stored {len(search_results)} search results in MongoDB")

        return search_results
    except Exception as e:
        print(f"Error querying recommendation system: {e}")
        return []
metadata ,image_urls=fetch_data_from_mongodb()
bm25, index = train_recommendation_system(metadata, image_urls)
if bm25 is None or index is None:
        print("Failed to train recommendation system. Exiting.")
        exit(1)

   
# save_model(bm25)

@app.post("/search")
async def search_products(request: SearchRequest):
    try:
        prompt = " ".join(request.tags)
        product_results = query_recommendation(
            prompt=prompt,
            image_url=request.image_url,
            bm25=bm25,
            index=index
        )
        # Extract only the 'productId' as strings
        product_ids = [product["productId"] for product in product_results]
        return SearchResponse(product_ids=product_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app (for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # metadata, image_urls = fetch_data_from_mongodb()
    # if metadata is None or image_urls is None:
    #     print("Failed to fetch data. Exiting.")
    #     exit(1)

   
   
    
    # sample_prompt = "i want an awesome chair made for children"
    # sample_image_url = "https://res.cloudinary.com/dxkqwj38h/image/upload/v1746186317/furniture_images/pvruqxtna2ncvmxvsylk.png"  # Replace with a valid Cloudinary URL
    # results = query_recommendation(sample_prompt, sample_image_url, bm25, index)

    # # Print results
    # print("hello")
    # for result in results:
    #  print(f"Product ID: {result['productId']}, Name: {result['name']}, Score: {result['score']}")


    

   