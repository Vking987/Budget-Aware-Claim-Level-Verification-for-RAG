from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


MODEL_NAME = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = f"hotpotqa_{MODEL_NAME.split('/')[-1]}"  # hotpotqa_bge-large-en-v1.5

TOP_K = 10


# -------------------------
# Load HotpotQA
# -------------------------

print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="train[:1000]"
)

print(f"Loaded {len(dataset)} questions")


# -------------------------
# Load embedding model
# -------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# -------------------------
# Connect to Qdrant
# -------------------------

client = QdrantClient(
    host="localhost",
    port=6333
)


# -------------------------
# Evaluation counters
# -------------------------

recall_1 = 0
recall_3 = 0
recall_5 = 0
recall_10 = 0

both_1 = 0
both_3 = 0
both_5 = 0
both_10 = 0


# -------------------------
# Evaluate each question
# -------------------------

for index, example in enumerate(dataset):

    question = example["question"]

    # HotpotQA stores supporting_facts as:
    #
    # {
    #     "title": [...],
    #     "sent_id": [...]
    # }
    #
    # We only need the supporting document titles here.

    gold_titles = set(
        example["supporting_facts"]["title"]
    )


    # -------------------------
    # Embed question
    # -------------------------

    query_vector = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()


    # -------------------------
    # Retrieve top 10
    # -------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K
    ).points


    retrieved_titles = [
        result.payload["title"]
        for result in results
    ]


    # Remove duplicate titles
    retrieved_titles = list(
        dict.fromkeys(retrieved_titles)
    )


    # -------------------------
    # Calculate Recall@K
    # -------------------------

    retrieved_at_1 = set(retrieved_titles[:1])
    retrieved_at_3 = set(retrieved_titles[:3])
    retrieved_at_5 = set(retrieved_titles[:5])
    retrieved_at_10 = set(retrieved_titles[:10])


    # At least one supporting document retrieved

    if gold_titles & retrieved_at_1:
        recall_1 += 1

    if gold_titles & retrieved_at_3:
        recall_3 += 1

    if gold_titles & retrieved_at_5:
        recall_5 += 1

    if gold_titles & retrieved_at_10:
        recall_10 += 1


    # -------------------------
    # Both supporting documents
    # -------------------------

    if gold_titles.issubset(retrieved_at_1):
        both_1 += 1

    if gold_titles.issubset(retrieved_at_3):
        both_3 += 1

    if gold_titles.issubset(retrieved_at_5):
        both_5 += 1

    if gold_titles.issubset(retrieved_at_10):
        both_10 += 1


    # -------------------------
    # Progress
    # -------------------------

    if (index + 1) % 100 == 0:
        print(f"Evaluated {index + 1}/{len(dataset)}")


# -------------------------
# Final results
# -------------------------

total = len(dataset)

print("\n==============================")
print("RETRIEVAL EVALUATION")
print("==============================")

print(f"\nRecall@1:  {recall_1 / total:.4f}")
print(f"Recall@3:  {recall_3 / total:.4f}")
print(f"Recall@5:  {recall_5 / total:.4f}")
print(f"Recall@10: {recall_10 / total:.4f}")

print("\nBoth supporting documents retrieved:")

print(f"Both@1:  {both_1 / total:.4f}")
print(f"Both@3:  {both_3 / total:.4f}")
print(f"Both@5:  {both_5 / total:.4f}")
print(f"Both@10: {both_10 / total:.4f}")