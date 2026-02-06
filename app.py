import os, base64
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from pathlib import Path

# Tooling
# from tools.tools import ()

# Directories
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
INCOMING_DIR = BASE_DIR / "incoming"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
PROCESSED_DIR = BASE_DIR / "processed_documents"
LOGS_DIR = BASE_DIR / "logs"
SETTINGS_DIR = BASE_DIR / "settings"

# URLs
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Read .env file and set the variables
load_dotenv()

# Retrive the credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID") or st.secrets["VECTOR_STORE_ID"]
MODEL_NAME = os.getenv("MODEL_NAME")

# Set the OpenAI API key in the os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
if not VECTOR_STORE_ID:
    st.warning("Vector store ID is not set. Please set the Vector Store ID in the environment")

# Configuration of the system promp
system_promp = """
You are a helpfull student assistant that can answer questions and help.
You are answering in 5-6 sentences only, providing meaningful information based on found documentation.
"""

# Store the previous response id
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User sidebar
with st.sidebar:
    st.header("User Controls")
    st.divider()

    # Clear the converstation history - reset chat history and context
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        # Reest the page
        st.rerun()

# Helper functions:
def build_input_parts(text: str,images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content = []
    if text and text.strip():
        content.append({
            "type" : "input_text",
            "text" : text.strip()
        })
    for img in images:
        content.append({
            "type" : "input_image",
            "image_url" : {"url" : img["data_url"]}
        })
    return [{"type": "message", "role": "user", "content": content}] if content else []

# Function to generate a response from the OpenAI reponses API
def call_responses_api(parts: List[Dict[str, Any]], previous_response_id: str = None):
    tools = [ 
        {
            "type" : "file_search", 
            "vector_store_ids" : [VECTOR_STORE_ID],
            "max_num_results" : 20
        }
    ]

    response = client.responses.create(
        model=MODEL_NAME,
        input = parts,
        instructions=system_promp,
        tools = tools,
        previous_response_id=previous_response_id
    )

    return response

# Function to get the text output
def get_text_output(response: Any) -> str:
    return response.output_text


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
## Upload images
uploaded = st.file_uploader(
    "Upload file(s)",
    type=["jpg","jpeg","png","webp","doc","docx","xls","xlsx","txt","pdf"],
    accept_multiple_files=True,
    key=f"file_uploader_{len(st.session_state.messages)}"
)

## Chat input
prompt = st.chat_input("Type your message here..")

if prompt is not None:
    # Process the images into an API-Compatible format
    images = []
    if uploaded:
        images = [
            {
                "mime_type" : f"image/{f.type.split('/')[-1]}" if f.type else "image/png",
                "data_url" : f"data:{f.type};base64,{base64.b64encode(f.read()).decode('utf-8')}"
            }
            for f in uploaded or []
        ]

    # Build the input parts for the reponses API
    parts = build_input_parts(prompt, images)

    # Store the messages
    st.session_state.messages.append(
        {
            "role" : "user",
            "content" : parts
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
                response = call_responses_api(parts, st.session_state.previous_response_id)
                output_text = get_text_output(response)

                # Display the AI's response
                st.markdown(output_text)
                #st.session_state.previous_response_id = response.id
                st.session_state.messages.append({"role" : "assistant", "content" : output_text })

                # Retrive the ID if available
                if hasattr(response, "id"):
                    st.session_state.previous_response_id = response.id
            except Exception as e:
                st.error(f"Error generating response: {e}")
                #st.session_state.previous_response_id = None