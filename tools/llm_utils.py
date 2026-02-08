def determine_category_llm(client, text, categories, model_name):
    """Determine document category using LLM."""
    try:
        categories_str = "\n".join(f"- {c}" for c in categories)
        prompt = (
            "You are a document classifier. Assign text to exactly one category. Respond with only the category name.\n"
            f"Categories:\n{categories_str}\n\nText:\n{text[:2000]}...\nCategory:"
        )
        response = client.responses.create(model="gpt-4.1-mini", input=prompt, temperature=0)
        category = response.output_text.strip()
        if category not in categories:
            return "Pozostałe dokumenty"
        return category
    except Exception:
        return "Pozostałe dokumenty"

def build_system_prompt(prompt_dict):
    """Flatten system prompt dict into formatted string."""
    parts = []
    for key, value in prompt_dict.items():
        if isinstance(value, dict):
            parts.append(f"{key.capitalize()}:")
            for subkey, subvalue in value.items():
                parts.append(f"- {subkey}: {subvalue}")
        else:
            parts.append(f"{key.capitalize()}: {value}")
    return "\n".join(parts)
