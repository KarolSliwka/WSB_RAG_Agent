def determine_category_llm(client, text, categories, model_name):
    """
    Determine document category using LLM.
    """
    try:
        categories_str = "\n".join(f"- {c}" for c in categories)
        prompt = (
            "You are a document classifier. Assign text to exactly one category. Respond with only the category name.\n"
            f"Categories:\n{categories_str}\n\nText:\n{text[:2000]}...\nCategory:"
        )
        response = client.responses.create(
            model=model_name,
            input=prompt,
            temperature=0.1,
            top_p = 0.85
        )
        category = response.output_text.strip()
        if category not in categories:
            return "Pozostałe dokumenty"
        return category
    except Exception:
        return "Pozostałe dokumenty"

def build_system_prompt(prompt_dict):
    """
    Flatten system prompt dict into formatted string.
    """
    parts = []
    for key, value in prompt_dict.items():
        if isinstance(value, dict):
            parts.append(f"{key.capitalize()}:")
            for subkey, subvalue in value.items():
                parts.append(f"- {subkey}: {subvalue}")
        else:
            parts.append(f"{key.capitalize()}: {value}")
    return "\n".join(parts)

def should_create_ticket(user_prompt: str, assistant_response: str) -> bool:
    """
    Decide whether to create a ticket.
    - True if user explicitly asks to create a ticket.
    - True if assistant suggests creating a ticket because it cannot answer confidently.
    """
    user_intent = "create ticket" in user_prompt.lower() or "open ticket" in user_prompt.lower()
    system_suggestion = "suggest creating a ticket" in assistant_response.lower() or "cannot answer" in assistant_response.lower()

    return user_intent or system_suggestion
