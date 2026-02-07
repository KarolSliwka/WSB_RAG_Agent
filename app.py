import os, base64
import streamlit as st
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Directories
BASE_DIR = Path(__file__).parent

## CSS file directory
CSS_FILE = BASE_DIR / "static" / "css" / "main.css"

## Documents directories
DOCS_DIR = BASE_DIR / "documents"
INCOMING_DIR = DOCS_DIR / "incoming"
KNOWLEDGE_DIR = DOCS_DIR / "knowledge"
PROCESSED_DIR = DOCS_DIR / "processed_documents"
LOGS_DIR = DOCS_DIR / "logs"

## Settings directory (json files)
SETTINGS_DIR = BASE_DIR / "settings"

# Read .env file and set the variables
load_dotenv()

# Retrive the credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
# VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID") or st.secrets["VECTOR_STORE_ID"]
MODEL_NAME = os.getenv("MODEL_NAME") or st.secrets["MODEL_NAME"]
EMBEDING_MODEL_NAME = os.getenv("EMBEDING_MODEL_NAME") or st.secrets["EMBEDING_MODEL_NAME"]
QDRANT_URL = os.getenv("QDRANT_URL") or st.secrets["QDRANT_URL"]
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or st.secrets["QDRANT_API_KEY"]

# Set the OpenAI API key in the os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Tooling
from tools.ticket import Ticket
from tools.tools import (
    build_input_parts,
    call_responses_api,
    get_text_output,
    import_and_index_documents_qdrant,
    get_qdrant_information_all
)

# URLs
#QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Initialize Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Load css file to override streamlit styles
@st.cache_data
def get_css(css_path: Path):
    with open(css_path) as f:
        return f.read()

st.markdown(f"<style>{get_css(CSS_FILE)}</style>", unsafe_allow_html=True)

# Load JSON files to objects
## -> LLM
with open(f"{SETTINGS_DIR}/llm.json", "r", encoding="utf-8") as file:
    llm_settings = json.load(file)
## System Prompt
system_prompt = " ".join(f"{key.capitalize()}: {value}" for key, value in llm_settings["system_prompt"].items())
## Developer promt
developer_prompt = " ".join(f"{key.capitalize()}: {value}" for key, value in llm_settings["developer_prompt"].items())

## -> Qdrant


## -> Models


## -> Categories
with open(f"{SETTINGS_DIR}/categories.json", "r", encoding="utf-8") as file:
    categories_list = json.load(file)

@dataclass
class Category:
    name: str
category_objects = [Category(name) for name in categories_list["categories"]]

# Get knowledge 

# Get Qdrant collections information
qdrant_collections_info = get_qdrant_information_all(QDRANT_URL, QDRANT_API_KEY)

# App configurations
st.set_page_config(
    page_title="WSB Agent",
    page_icon=":material/chat_bubble:", # speech bubble icon
    layout="centered"
)

# Add title to the app
st.title("WSB AI Agent")

# Add a description to the ap
st.markdown("**Your inteligent WSB Assistant powered by GPT-5 and RAG technology**")
st.divider()

# Add a collapsible section
with st.expander("About this chat", expanded=False):
    st.markdown(
        """
        ### WSB Intelligent Assistant (Chat & Ticketing)
        - Model
            - **GPT-5**  via **OpenAI Responses API**
        - Retrieval (RAG)
            - **File Search Tool**  Using a pre-built **Vector Store** to ground answers in your documents.
        - Features
            - Multi-turn conversational chat
            - Document & image input
            - Clear / reset conversation
        - Secrets & Configuration Reads the following from **Streamlit secrets** or environment variables:
            - `OPENAI_API_KEY`
            - `VECTOR_STORE_ID`
        ---
        - How it works
            1. Your message (and optional document or image) is sent to the **Responses API**
            2. Relevant passages are retrieved from the **Vector Store**
            3. The model uses this context to generate a grounded, accurate response
        """
    )

# Initialize the OpenAi Client
client = OpenAI()

# Warn if OpenAI API Key or the VectorStoreId are not set
if not OPENAI_API_KEY:
    st.warning("OpenAI API Key is not set. Please set the OpenAi API key in the environment")
# if not VECTOR_STORE_ID:
#     st.warning("Vector store ID is not set. Please set the Vector Store ID in the environment")

# Store the previous response id
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User sidebar
with st.sidebar:
    st.header("User Controls")

    # Clear the converstation history - reset chat history and context
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        # Reest the page
        st.rerun()

    st.divider()

    # Qdrant Management
    st.subheader("Qdrant Management")
    if st.button(f"Import 'Knowldege' Documents {knowledge_not_imported}"):
        import_and_index_documents_qdrant(qdrant_client, client, EMBEDING_MODEL_NAME, KNOWLEDGE_DIR, MODEL_NAME, SETTINGS_DIR)

        # Display each collection
        for coll in qdrant_collections_info:
            st.markdown(f"**Collection:** {coll['name']}")
            st.markdown(f"- Points: {coll['points']}")
            st.markdown(f"- Segments: {coll['segments']}")
            st.markdown(f"- Size: {round(coll['size_bytes'] / (1024*1024), 2)} MB")
            st.markdown(f"- Distance: {coll['distance']}")
            st.divider()

# Render all previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        # extract content from the message structure
        if isinstance(m["content"], list):
            for part in m["content"]:
                for content_item in part.get("content", []):
                    if content_item.get("type") == "input_text":
                        st.markdown(content_item["text"])
                    elif content_item.get("type") == "input_image":
                        st.image(content_item["image_url"], width=100)
        elif isinstance(m["content"], str):
            st.markdown(m["content"])

# User Interface
## Upload files
uploaded = st.file_uploader(
    "Upload file(s)",
    type=["jpg","jpeg","png","webp","doc","docx","xls","xlsx","txt","pdf"],
    accept_multiple_files=True,
    key=f"file_uploader_{len(st.session_state.messages)}"
)

# Zapis wszystkich uploadów w sesji
if uploaded:
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    st.session_state["uploaded_files"].extend(uploaded)

## Chat input
prompt = st.chat_input("Type your message here..")

if prompt is not None:
    # Rozdziel pliki na obrazy i pozostałe dokumenty
    images = []
    documents = []
    for f in uploaded or []:
        f.seek(0)
        if f.type and f.type.startswith("image"):
            images.append({
                "mime_type": f"type/{f.type.split('/')[-1]}",
                "data_url": f"data:{f.type};base64,{base64.b64encode(f.read()).decode('utf-8')}"
            })
        else:
            documents.append(f)  # wszystkie pozostałe pliki jako obiekty

    # Build the input parts for the responses API (tylko obrazy)
    parts = build_input_parts(prompt, images)

    # Store the messages
    st.session_state.messages.append(
        {
            "role": "user",
            "content": parts
        }
    )

    # Display the user's message
    with st.chat_message("user"):
        for p in parts:
            if p["type"] == "message":
                for content_item in p.get("content", []):
                    if content_item["type"] == "input_text":
                        st.markdown(content_item["text"])
                    elif content_item["type"] == "input_image":
                        st.image(content_item["image_url"], width=100)

    # Generate the AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking.."):
            try:
                # Call the refactored function using Qdrant
                response = call_responses_api(
                    system_prompt=system_prompt,
                    client=client,
                    model_name=MODEL_NAME,
                    parts=parts,
                    qdrant_client=qdrant_client,
                    qdrant_collection="Documents",  # replace with your collection name
                    top_k=5,
                    previous_response_id=st.session_state.previous_response_id
                )

                output_text = get_text_output(response)

                # Display the AI's response
                st.markdown(output_text)

                # Store assistant message
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                # Retrieve the ID if available
                if hasattr(response, "id"):
                    st.session_state.previous_response_id = response.id
            except Exception as e:
                st.error(f"Error generating response: {e}")