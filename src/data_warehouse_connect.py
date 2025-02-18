"""
    handle database interactions separately
"""

import pymongo
from config import MONGO_URI


def connect_mongo():
    """Connects to MongoDB and returns the database."""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client["general_data"]
        print("connected")
        return db
    except Exception as e:
        print("connection failed")
        return None
    
db = connect_mongo()
