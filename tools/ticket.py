from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import uuid
from openai import OpenAI
import json
from pathlib import Path

# Ścieżki do plików settings
BASE_DIR = Path(__file__).parent.parent
CATEGORIES_FILE = BASE_DIR / "settings" / "categories.json"
PRIORITIES_FILE = BASE_DIR / "settings" / "priorities.json"

# Wczytaj kategorie i priorytety
with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
    categories_list = json.load(f)["categories"]

with open(PRIORITIES_FILE, "r", encoding="utf-8") as f:
    priorities_list = json.load(f)["priorities"]

@dataclass
class Ticket:
    """
    Klasa reprezentująca zgłoszenie do BOS (Biuro Obsługi Studenta)
    """
    title: str
    description: str
    created_by: str
    category: str = "Pozostałe dokumenty"
    priority: str = "Informacyjne"
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    ticket_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "Open"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_attachment(self, file_url: str, file_name: str):
        self.attachments.append({"file_url": file_url, "file_name": file_name})
        self.updated_at = datetime.now().isoformat()

    def update_status(self, new_status: str):
        self.status = new_status
        self.updated_at = datetime.now().isoformat()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "title": self.title,
            "description": self.description,
            "created_by": self.created_by,
            "category": self.category,
            "priority": self.priority,
            "attachments": self.attachments,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


def assign_category_and_priority(client: OpenAI, model_name, conversation_context: str) -> Dict[str, str]:
    """
    Funkcja prosi LLM o wybranie odpowiedniej kategorii i priorytetu
    na podstawie kontekstu rozmowy.
    """
    system_prompt = f"""
    Jesteś asystentem do tworzenia ticketów BOS (Biuro Obsługi Studenta).
    Na podstawie treści zgłoszenia użytkownika wybierz:
    1. Kategorie z listy: {categories_list}
    2. Priorytet z listy: {priorities_list}
    
    Odpowiedz w formacie JSON:
    {{
        "category": "<wybrana_kategoria>",
        "priority": "<wybrany_priorytet>"
    }}
    """

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": conversation_context}
                ]
            }
        ],
        instructions=system_prompt
    )

    # Odczytaj JSON z odpowiedzi
    import json
    try:
        output_json = json.loads(response.output_text)
        category = output_json.get("category", "Pozostałe dokumenty")
        priority = output_json.get("priority", "Informacyjne")
    except Exception:
        category = "Pozostałe dokumenty"
        priority = "Informacyjne"

    return {"category": category, "priority": priority}