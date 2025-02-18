"""
Configuration file for the project
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch values from .env
MONGO_URI = os.getenv("MONGO_URI")