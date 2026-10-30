from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from tqdm import tqdm


# -------------------------
# Configuration
# -------------------------

NUM_EXAMPLES = 1000
BATCH_SIZE = 64

# -------------------------
# Load HotpotQA
# -------------------------

print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"train[:{NUM_EXAMPLES}]"
)

print(f"Loaded {len(dataset)} questions")


# -------------------------
# Load embedding model
# -------------------------

print("Loading embedding model...")

MODEL_NAME = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = f"hotpotqa_{MODEL_NAME.split('/')[-1]}"  # hotpotqa_bge-large-en-v1.5

model = SentenceTransformer(MODEL_NAME)
vector_size = model.get_embedding_dimension()   

# -------------------------
# Connect to Qdrant
# -------------------------

client = QdrantClient(
    host="localhost",
    port=6333
)

# Recreate collection
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )



# -------------------------
# Prepare documents
# -------------------------

documents = []

point_id = 0

for example in tqdm(dataset, desc="Preparing documents"):

    hotpot_id = example["id"]

    for title, sentences in zip(
    example["context"]["title"],
    example["context"]["sentences"]
):

        text = " ".join(sentences)

        documents.append({
            "id": point_id,
            "text": f"Title: {title}\n\n{text}",
            "title": title,
            "hotpot_id": hotpot_id
        })

        point_id += 1


print(f"Prepared {len(documents)} documents")


# -------------------------
# Embed + upload
# -------------------------

for start in tqdm(
    range(0, len(documents), BATCH_SIZE),
    desc="Embedding and uploading"
):

    batch = documents[start:start + BATCH_SIZE]

    texts = [doc["text"] for doc in batch]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    points = []

    for doc, embedding in zip(batch, embeddings):

        points.append(
            PointStruct(
                id=doc["id"],
                vector=embedding.tolist(),
                payload={
                    "text": doc["text"],
                    "title": doc["title"],
                    "hotpot_id": doc["hotpot_id"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


print("\nDone!")
print(f"Collection: {COLLECTION_NAME}")
print(f"Documents stored: {len(documents)}")