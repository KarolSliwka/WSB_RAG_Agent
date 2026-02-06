#%pip install pandas openai pydantic dotenv 

import os, base64
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from IPython.display import Markdown, display
from dotenv import load_dotenv

# directories
directory = "/Users/karolsliwka/Desktop/AICourse/agent2_tickets/documents/knowledge"
os.chdir(directory)

# read .env file
load_dotenv(".env")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Define the configurations
client = OpenAI(
    api_key = API_KEY
)