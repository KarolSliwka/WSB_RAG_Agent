import json
import uuid
import hashlib
import pytesseract
import docx
import shutil
import pdfplumber
import streamlit as st
from pdf2image import convert_from_path
from pathlib import Path
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from .api_utils import embed_text, embed_image
from .llm_utils import determine_category_llm
from .ticket import Ticket

# Ensure Tesseract binary is found
TESSERACT_PATH = shutil.which("tesseract")
if TESSERACT_PATH is None:
    raise RuntimeError("Tesseract OCR not found in PATH")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Set languages you want OCR to support => english / polish
OCR_LANGS = "eng+pol"
OCR_CONFIG = "--oem 3 --psm 6"

def ensure_collections_exist(qdrant_client, collections=None, vector_size=1536, distance="Cosine"):
    """
    Ensures that the specified Qdrant collections exist. 
    Creates missing collections; leaves existing ones untouched.

    Args:
        qdrant_client: QdrantClient instance.
        collections: List of collection names to ensure exist.
        vector_size: Dimensionality of vector embeddings.
        distance: Distance metric to use ("Cosine", "Euclid", "Dot").

    Returns:
        dict: {"created": [...], "already_exist": [...]}
    """
    if collections is None:
        collections = ["Documents", "Knowledge", "Processed", "Tickets"]

    # Get current collections in Qdrant
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]

    created_collections = []
    already_exist = []

    for collection_name in collections:
        if collection_name not in existing_collections:
            print(f"[INFO] Creating missing collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            created_collections.append(collection_name)
        else:
            print(f"[INFO] Collection already exists: {collection_name}")
            already_exist.append(collection_name)

    return {"created": created_collections, "already_exist": already_exist}


def file_hash(file_path):
    """
    Compute SHA256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def not_imported_files(qdrant_client, knowledge_dir, collection_name="Documents"):
    """
    Returns the number of files in a directory not yet imported into a Qdrant collection,
    using file hashes for accurate deduplication across sources.
    """
    knowledge_dir = Path(knowledge_dir)

    # Fetch all points from Qdrant
    existing_points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10000)
    imported_hashes = {p.payload.get("hash") for p in existing_points if "hash" in p.payload}

    # Compute hashes for all local files
    all_hashes = {file_hash(f) for f in knowledge_dir.glob("**/*") if f.is_file()}

    # Files not yet imported
    not_imported = all_hashes - imported_hashes
    return len(not_imported)

def get_qdrant_collection_summary(qdrant_url, qdrant_api_key):
    """
    Fetch summary info for all Qdrant collections and sort in custom order:
    Documents, Knowledge, Processed, Tickets
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

        # Custom sort order
        order = ["Documents", "Knowledge", "Processed", "Tickets"]
        summary_list.sort(key=lambda x: order.index(x["name"]) if x["name"] in order else len(order))

        return summary_list

    except Exception as e:
        print(f"Error fetching Qdrant info: {e}")
        return []

def import_and_index_documents_qdrant(qdrant_client, client, embedding_model_name, knowledge_dir, model_name, settings_dir, batch_size=200):
    """
    Import and index documents into Qdrant collection with Streamlit progress display.
    """
    knowledge_dir = Path(knowledge_dir)
    collection_name = "Documents"

    # Load categories
    with open(Path(settings_dir) / "categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f).get("categories", [])

    # Ensure collection exists
    if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance="Cosine")
        )

    # Fetch existing hashes to avoid duplicates
    existing_points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10000)
    imported_hashes = {p.payload.get("hash") for p in existing_points if "hash" in p.payload}

    points_to_upload = []

    progress_text = st.empty()  # For status updates
    progress_bar = st.progress(0)
    total_files = len(list(knowledge_dir.glob("**/*")))
    processed_files = 0

    def flush_points():
        """Upload current batch and clear list"""
        if points_to_upload:
            for i in range(0, len(points_to_upload), batch_size):
                batch = points_to_upload[i:i+batch_size]
                qdrant_client.upsert(collection_name=collection_name, points=batch)
            points_to_upload.clear()

    for file_path in knowledge_dir.glob("**/*"):
        if not file_path.is_file():
            continue

        file_h = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if file_h in imported_hashes:
            progress_text.write(f"✅ [SKIP] Already imported: {file_path.name}")
            processed_files += 1
            progress_bar.progress(processed_files / total_files)
            continue

        progress_text.write(f"⏳ [PROCESSING] {file_path.name}")

        text = ""
        try:
            suffix = file_path.suffix.lower()
            if suffix in [".txt", ".md"]:
                text = file_path.read_text(encoding="utf-8")
            elif suffix == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif suffix == ".docx":
                import docx
                doc = docx.Document(str(file_path))
                text = "\n".join([p.text for p in doc.paragraphs])
            elif suffix in [".png", ".jpg", ".jpeg", ".webp"]:
                embedding = embed_image(client, embedding_model_name, file_path)
                points_to_upload.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "source": str(file_path),
                            "type": "image",
                            "hash": file_h,
                            "upload_timestamp": datetime.utcnow().isoformat()
                        }
                    )
                )
                flush_points()
                progress_text.write(f"✅ [UPLOADED] Image: {file_path.name}")
                processed_files += 1
                progress_bar.progress(processed_files / total_files)
                continue
            else:
                progress_text.write(f"⚠️ [SKIP] Unsupported file type: {file_path.name}")
                processed_files += 1
                progress_bar.progress(processed_files / total_files)
                continue
        except Exception as e:
            progress_text.write(f"❌ [ERROR] Failed to read {file_path.name}: {e}")
            processed_files += 1
            progress_bar.progress(processed_files / total_files)
            continue

        if not text.strip():
            progress_text.write(f"⚠️ [WARN] No text extracted from {file_path.name}, skipping")
            processed_files += 1
            progress_bar.progress(processed_files / total_files)
            continue

        try:
            category = determine_category_llm(client, text, categories, model_name)
        except Exception:
            category = "Pozostałe dokumenty"

        # Chunk text and embed
        chunk_size = 800
        overlap = 200
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            start += chunk_size - overlap

            try:
                embedding = embed_text(client, embedding_model_name, chunk_text)
            except Exception as e:
                progress_text.write(f"❌ [ERROR] Failed to embed chunk: {e}")
                continue

            points_to_upload.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": str(file_path),
                        "type": "document",
                        "hash": file_h,
                        "category": category,
                        "chunk_start": start,
                        "chunk_end": end,
                        "text": chunk_text,
                        "upload_timestamp": datetime.utcnow().isoformat()
                    }
                )
            )

            if len(points_to_upload) >= batch_size:
                flush_points()

        processed_files += 1
        progress_bar.progress(processed_files / total_files)
        progress_text.write(f"✅ [UPLOADED] {file_path.name}")

    flush_points()
    progress_text.write(f"🎉 Finished uploading points to {collection_name}")


def extract_text_from_pdf(file_path):
    """
    Robust PDF text extraction:
    - Falls back to OCR for image-based pages with multiple languages
    """
    file_path = Path(file_path)
    full_text = []

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()

                if page_text and page_text.strip():
                    full_text.append(page_text)
                else:
                    # Page is likely scanned → OCR
                    print(f"[OCR] Page {i+1} might be image-based, performing OCR")
                    try:
                        images = convert_from_path(str(file_path), first_page=i+1, last_page=i+1, dpi=300)
                        for img in images:
                            ocr_text = pytesseract.image_to_string(img, lang=OCR_LANGS, config=OCR_CONFIG)
                            full_text.append(ocr_text)
                    except Exception as ocr_err:
                        print(f"[ERROR] OCR failed on page {i+1} of {file_path}: {ocr_err}")
    except Exception as e:
        print(f"[ERROR] Failed to open PDF {file_path}: {e}")
        return ""

    # Normalize line breaks and remove empty lines
    combined_text = "\n".join([line.strip() for line in "\n".join(full_text).splitlines() if line.strip()])
    return combined_text


def create_ticket_in_qdrant(qdrant_client, ticket_data: dict, embedding_vector=None):
    """
    Create a ticket in the Qdrant 'Tickets' collection.
    Only creates a ticket if all required fields are present.

    Args:
        qdrant_client: QdrantClient instance
        ticket_data: dict with keys:
            - first_name
            - last_name
            - email
            - index_number
            - description
            - title (optional)
            - priority (optional, default 'Medium')
        embedding_vector: optional vector for semantic search

    Returns:
        str: ticket_id of the created ticket

    Raises:
        ValueError: if required fields are missing
    """
    REQUIRED_FIELDS = ["first_name", "last_name", "email", "index_number", "description"]
    missing = [f for f in REQUIRED_FIELDS if not ticket_data.get(f)]
    if missing:
        raise ValueError(f"Nie można utworzyć ticketu. Brakujące dane: {', '.join(missing)}")

    # Create Ticket object
    ticket = Ticket(
        title=ticket_data.get("title", ticket_data["description"][:50] + "..."),
        description=ticket_data["description"],
        created_by=f"{ticket_data['first_name']} {ticket_data['last_name']}",
        priority=ticket_data.get("priority", "Medium")
    )

    # Use embedding vector if provided, else default zero vector
    vector = embedding_vector if embedding_vector else [0.0]*1536

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=ticket.as_dict()
    )

    qdrant_client.upsert(collection_name="Tickets", points=[point])
    return ticket.ticket_id