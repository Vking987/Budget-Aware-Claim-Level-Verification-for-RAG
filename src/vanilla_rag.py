from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI


# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = f"hotpotqa_{MODEL_NAME.split('/')[-1]}"  
TOP_K = 5

LLAMA_BASE_URL = "http://127.0.0.1:8080/v1"


# ==================================================
# Load embedding model
# ==================================================

print("Loading BGE-large...")

embedding_model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)


# ==================================================
# Connect to Qdrant
# ==================================================

qdrant = QdrantClient(
    host="localhost",
    port=6333
)


# ==================================================
# Connect to llama.cpp
# ==================================================

llm = OpenAI(
    base_url=LLAMA_BASE_URL,
    api_key="local"
)


# ==================================================
# Find the model served by llama.cpp
# ==================================================

models = llm.models.list()

model_name = models.data[0].id

print(f"Llama model: {model_name}")


# ==================================================
# Retrieval function
# ==================================================

def retrieve(question, top_k=TOP_K):

    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    return results


# ==================================================
# Generate answer
# ==================================================

def generate_answer(question, results):

    context_parts = []

    for i, result in enumerate(results):

        text = result.payload["text"]

        context_parts.append(
            f"[Document {i + 1}]\n{text}"
        )

    context = "\n\n".join(context_parts)


    prompt = f"""
You are a question-answering assistant.

Answer the question using only the provided context.

If the context does not contain enough information to answer
the question, say that the information is insufficient.

Do not invent facts.

Context:

{context}

Question:

{question}

Answer:
"""


    response = llm.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=256
    )

    return response.choices[0].message.content


# ==================================================
# Test with HotpotQA
# ==================================================

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="train[:5]"
)


for i, example in enumerate(dataset):

    question = example["question"]

    print("\n")
    print("=" * 70)
    print(f"QUESTION {i + 1}")
    print("=" * 70)

    print(question)


    # ----------------------------------------------
    # Retrieval
    # ----------------------------------------------

    results = retrieve(question)


    print("\nRETRIEVED DOCUMENTS")
    print("-" * 70)

    for rank, result in enumerate(results):

        print(
            f"{rank + 1}. "
            f"{result.payload['title']} "
            f"(score={result.score:.4f})"
        )


    # ----------------------------------------------
    # Generation
    # ----------------------------------------------

    answer = generate_answer(
        question,
        results
    )


    print("\nLLAMA ANSWER")
    print("-" * 70)

    print(answer)


    print("\nGROUND TRUTH")
    print("-" * 70)

    print(example["answer"])