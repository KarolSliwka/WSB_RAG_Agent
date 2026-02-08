from typing import Any, List, Dict
from .llm_utils import build_system_prompt

def build_input_parts(text: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert user text and images into chat input parts."""
    content = []
    if text and text.strip():
        content.append({"type": "input_text", "text": text.strip()})
    for img in images:
        content.append({"type": "input_image", "image_url": {"url": img["data_url"]}})
    return [{"type": "message", "role": "user", "content": content}] if content else []

def call_responses_api(system_prompt: str, client, model_name: str, embedding_model_name: str, parts: List[Dict[str, Any]], qdrant_client=None, qdrant_collection=None, top_k=5, previous_response_id=None):
    """Generate AI response, optionally using Qdrant for context."""
    user_texts = [c["text"] for p in parts for c in p.get("content", []) if c.get("type") == "input_text"]
    user_prompt = "\n".join(user_texts)
    context_texts = []
    if qdrant_client and qdrant_collection:
        embedding = client.embeddings.create(model=embedding_model_name, input=user_prompt).data[0].embedding
        search_result = qdrant_client.query_points(collection_name=qdrant_collection, query=embedding, limit=top_k, with_payload=True).points
        context_texts = [hit.payload.get("text", "") for hit in search_result if hit.payload]
    final_input = f"{system_prompt}\n\nContext:\n{'\n\n'.join(context_texts)}\n\nUser: {user_prompt}"
    return client.responses.create(model=model_name, input=final_input, previous_response_id=previous_response_id)

def get_text_output(response: Any) -> str:
    """Extract text output from API response."""
    return response.output_text

def embed_text(client, embedding_model_name, text: str) -> List[float]:
    """Generate vector embedding for text."""
    response = client.embeddings.create(model=embedding_model_name, input=text)
    return response.data[0].embedding

def embed_image(client, embedding_model_name, image_path) -> List[float]:
    """Embed image as pseudo-text using filename."""
    from .api_utils import embed_text
    return embed_text(client, embedding_model_name, f"Image: {image_path.name}")
