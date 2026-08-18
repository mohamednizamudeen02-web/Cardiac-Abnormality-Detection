import os
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

load_dotenv()

MONGO_URI = os.environ.get('MONGO_URI')
DB_NAME = os.environ.get('MONGO_DB', 'cardiac_monitor')

print(f"Connecting to: {MONGO_URI}")
print(f"Database: {DB_NAME}")

try:
    import ssl
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tls=True,
        tlsAllowInvalidCertificates=True,
        tlsAllowInvalidHostnames=True,
        retryWrites=True
    )
    db = client[DB_NAME]
    
    # Check connection
    client.admin.command('ping')
    print("[OK] Successfully connected to MongoDB Atlas!")
    
    # List collections
    collections = db.list_collection_names()
    print(f"Collections in {DB_NAME}: {collections}")
    
    if 'users' in collections:
        users = list(db.users.find({}, {'_id': 0, 'password': 0}))
        print(f"Found {len(users)} users in 'users' collection:")
        for u in users:
            print(f" - {u.get('username')} ({u.get('role')})")
    else:
        print("[FAILED] 'users' collection does not exist!")

    # Check other collections to see if ANY data is there
    for col_name in ['patients', 'ecg_analyses']:
        if col_name in collections:
            count = db[col_name].count_documents({})
            print(f" - {col_name}: {count} documents")

except Exception as e:
    print(f"[FAILED] Error: {e}")
