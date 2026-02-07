import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

# Helper functions
def build_input_parts(text: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content = []
    if text and text.strip():
        content.append({
            "type": "input_text",
            "text": text.strip()
        })
    for img in images:
        content.append({
            "type": "input_image",
            "image_url": {"url": img["data_url"]}
        })
    return [{"type": "message", "role": "user", "content": content}] if content else []

## Function to generate a response from OpenAI Responses API using Qdrant
def call_responses_api(
    system_prompt: str,
    client,
    model_name: str,
    parts: List[Dict[str, Any]],
    qdrant_client: QdrantClient = None,
    qdrant_collection: str = None,
    top_k: int = 5,
    previous_response_id: str = None
    ):
    """
    Generate a response from OpenAI Responses API.
    Optionally uses Qdrant to retrieve relevant context.
    """

    # Extract user text from parts
    user_texts = []
    for part in parts:
        for content_item in part.get("content", []):
            if content_item.get("type") == "input_text":
                user_texts.append(content_item["text"])
    user_prompt = "\n".join(user_texts)

    # Retrieve context from Qdrant if provided
    context_texts = []
    if qdrant_client and qdrant_collection:
        embedding = client.embeddings.create(
            model=os.getenv("EMBEDING_MODEL_NAME"),
            input=user_prompt
        ).data[0].embedding

        search_result = qdrant_client.query_points(
            collection_name=qdrant_collection,
            query=embedding,
            limit=top_k,
            with_payload=True
        ).points

        context_texts = [
            hit.payload.get("text", "")
            for hit in search_result
            if hit.payload
        ]

    # Combine system prompt, context, and user input
    final_input = f"{system_prompt}\n\nContext:\n{'\n\n'.join(context_texts)}\n\nUser: {user_prompt}"

    # Call OpenAI Responses API
    response = client.responses.create(
        model=model_name,
        input=final_input,
        previous_response_id=previous_response_id
    )

    return response

## Function to get the text output
def get_text_output(response: Any) -> str:
    return response.output_text

# Qdrant functions
def get_qdrant_information_all(qdrant_url, qdrant_api_key):
    try:
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections_info = []

        # Get collection descriptions
        collections_list = qdrant.get_collections().collections

        for coll_desc in collections_list:
            # CollectionDescription has .name
            collection_name = coll_desc.name

            # Get detailed info
            coll_info = qdrant.get_collection(collection_name=collection_name)

            # Depending on version, coll_info.vectors may be a dict
            vectors_count = 0
            if hasattr(coll_info, "vectors_count"):
                vectors_count = coll_info.vectors_count
            elif hasattr(coll_info, "vectors") and isinstance(coll_info.vectors, dict):
                vectors_count = sum(vectors_count for v in coll_info.vectors.values())

            # Shards / segments
            segments = len(coll_info.shards) if hasattr(coll_info, "shards") else 0

            # Distance / metric
            distance = getattr(coll_info, "distance", "unknown")

            # Storage size
            size_bytes = getattr(coll_info, "storage_size", 0)

            collections_info.append({
                "name": collection_name,
                "points": vectors_count,
                "segments": segments,
                "size_bytes": size_bytes,
                "distance": distance
            })

        return collections_info

    except Exception as e:
        print(f"Error fetching Qdrant info: {e}")
        return []


# Import and index documents in qdrant
def import_and_index_documents_qdrant(qdrant_client, client, embedding_model_name, knowledge_dir, model_name, settings_dir):
    knowledge_dir = Path(knowledge_dir)
    collection_name = "Documents"

    # Load categories
    with open(f"{settings_dir}/categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f)["categories"]

    # Ensure collection exists
    if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance="Cosine")
        )

    points_to_upload = []

    for file_path in knowledge_dir.glob("**/*"):
        if not file_path.is_file():
            continue

        try:
            if file_path.suffix.lower() in [".txt", ".md"]:
                text = file_path.read_text(encoding="utf-8")
            elif file_path.suffix.lower() == ".pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(str(file_path))
                text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            elif file_path.suffix.lower() == ".docx":
                import docx
                doc = docx.Document(str(file_path))
                text = "\n".join([p.text for p in doc.paragraphs])
            elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                # For images, embed without category
                embedding = embed_image(client, embedding_model_name, file_path)
                points_to_upload.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "source": str(file_path),
                            "type": "image",
                            "upload_timestamp": datetime.utcnow().isoformat()
                        }
                    )
                )
                continue
            else:
                continue
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

        if not text.strip():
            print(f"[SKIP] {file_path} has no readable text")
            continue
        else:
            print(f"[READ] {file_path} length={len(text)}")

        # Determine category for the document
        try:
            category = determine_category_llm(client, text, categories, model_name)
        except Exception as e:
            print(f"[WARN] Failed to determine category for {file_path}: {e}")
            category = "Pozostałe dokumenty"

        # Chunk text
        chunk_size = 800
        overlap = 200
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            embedding = embed_text(client, embedding_model_name, chunk_text)
            points_to_upload.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk_text,
                        "source": str(file_path),
                        "category": category,
                        "upload_timestamp": datetime.utcnow().isoformat()
                    }
                )
            )
            if end == len(text):
                break
            start += chunk_size - overlap

    # Sanity check
    assert all(
        isinstance(p.id, str) and len(p.vector) == 1536
        for p in points_to_upload
    ), "Invalid point detected (bad ID or vector size)"

    # Upload points in batches
    batch_size = 20
    for i in range(0, len(points_to_upload), batch_size):
        qdrant_client.upsert(collection_name=collection_name, points=points_to_upload[i:i+batch_size])

    print(f"Indexed {len(points_to_upload)} points into Qdrant collection '{collection_name}'")

# Embedding text
def embed_text(client, embedding_model_name, text: str) -> List[float]:
    """Generate vector embedding using OpenAI text-embedding-3-small."""
    response = client.embeddings.create(
        model=embedding_model_name,
        input=text
    )
    return response.data[0].embedding

# Embedding images
def embed_image(client, embedding_model_name, image_path) -> List[float]:
    """Embed image using a textual proxy (filename)."""
    # For a proper embedding, you could use OpenAI CLIP or another image embedding model
    # Here we just use a pseudo-text embedding of the filename
    return embed_text(client, embedding_model_name, f"Image: {image_path.name}")

# Function that will determine file category
def determine_category_llm(client, text, categories, model_name):
    """
    Determines the most appropriate category for a given text using the OpenAI Responses API.

    Args:
        text (str): The document content.
        categories (list): List of available categories (strings).
        client (OpenAI): An initialized OpenAI client.
    
    Returns:
        str: The selected category (exactly one from categories).
    """
    try:
        categories_str = "\n".join(f"- {c}" for c in categories)
        prompt = (
            "You are a document classifier. "
            "Assign the text below to exactly one of the following categories. "
            "Respond with only the category name exactly as listed.\n\n"
            f"Categories:\n{categories_str}\n\n"
            f"Text:\n{text[:2000]}...\n\n"
            "Category:"
        )

        # Use Responses API
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0
        )

        # Extract text output
        category = response.output_text.strip()

        # Validate category
        if category not in categories:
            print(f"[WARN] LLM returned invalid category: '{category}'. Using fallback.")
            return "Pozostałe dokumenty"

        return category

    except Exception as e:
        print(f"[ERROR] determine_category_llm failed: {e}")
        return "Pozostałe dokumenty"