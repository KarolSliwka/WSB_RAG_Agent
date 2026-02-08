import json
import uuid
from pathlib import Path
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from .api_utils import embed_text, embed_image
from .llm_utils import determine_category_llm

def not_imported_files(qdrant_client, knowledge_dir, collection_name="Documents"):
    """
    Returns the number of files in a directory not yet imported into a Qdrant collection.
    """
    knowledge_dir = Path(knowledge_dir)
    all_files = {str(f) for f in knowledge_dir.glob("**/*") if f.is_file()}
    existing_points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10000)
    imported_files = {p.payload.get("source") for p in existing_points if "source" in p.payload}
    return len(all_files - imported_files)

def get_qdrant_collection_summary(qdrant_url, qdrant_api_key):
    """
    Fetch summary info for all Qdrant collections.
    """
    try:
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        summary_list = []
        for coll_desc in qdrant.get_collections().collections:
            collection_name = coll_desc.name
            points = qdrant.count(collection_name=collection_name).count
            coll_info = qdrant.get_collection(collection_name=collection_name)
            status = getattr(coll_info, "status", "GREEN")
            replication_factor = getattr(coll_info, "replication_factor", 1)
            shards = getattr(coll_info, "shard_number", len(getattr(coll_info, "shards", [])))
            vector_field = "Default"
            vector_size = 0
            distance = "unknown"
            if hasattr(coll_info, "vectors") and isinstance(coll_info.vectors, dict):
                vector_field = list(coll_info.vectors.keys())[0]
                vector_config = list(coll_info.vectors.values())[0]
                vector_size = getattr(vector_config, "size", 0)
                distance = getattr(vector_config, "distance", "unknown")
            summary_list.append({
                "name": collection_name,
                "status": status,
                "points": points,
                "shards": shards,
                "replicas": replication_factor,
                "vector_field": vector_field,
                "vector_size": vector_size,
                "distance": distance
            })
        return summary_list
    except Exception as e:
        print(f"Error fetching Qdrant info: {e}")
        return []

def import_and_index_documents_qdrant(qdrant_client, client, embedding_model_name, knowledge_dir, model_name, settings_dir):
    """
    Import and index documents into Qdrant collection.
    """
    knowledge_dir = Path(knowledge_dir)
    collection_name = "Documents"
    with open(f"{settings_dir}/categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f)["categories"]

    if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance="Cosine")
        )

    existing_points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10000)
    imported_files = set(p.payload.get("source") for p in existing_points if "source" in p.payload)
    points_to_upload = []

    for file_path in knowledge_dir.glob("**/*"):
        if not file_path.is_file() or str(file_path) in imported_files:
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
                embedding = embed_image(client, embedding_model_name, file_path)
                points_to_upload.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={"source": str(file_path), "type": "image", "upload_timestamp": datetime.utcnow().isoformat()}
                    )
                )
                continue
            else:
                continue
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            continue

        if not text.strip():
            continue

        try:
            category = determine_category_llm(client, text, categories, model_name)
        except Exception:
            category = "Pozostałe dokumenty"

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
                    payload={"text": chunk_text, "source": str(file_path), "category": category, "upload_timestamp": datetime.utcnow().isoformat()}
                )
            )
            if end == len(text):
                break
            start += chunk_size - overlap

    batch_size = 20
    for i in range(0, len(points_to_upload), batch_size):
        qdrant_client.upsert(collection_name=collection_name, points=points_to_upload[i:i+batch_size])
    print(f"Indexed {len(points_to_upload)} new points into '{collection_name}'")
