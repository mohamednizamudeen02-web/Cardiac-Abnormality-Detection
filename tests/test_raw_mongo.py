import os
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
import time

load_dotenv()
uri = "mongodb://smuntasir2005_db_user:newnew%4012345@cluster0-shard-00-00.9mih2vm.mongodb.net:27017/?authSource=admin&ssl=true"

print(f"Testing direct connection to single Shard...")
try:
    start = time.time()
    client = MongoClient(
        uri, 
        directConnection=True,
        tlsAllowInvalidCertificates=True,
        tlsAllowInvalidHostnames=True,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000
    )
    # Trigger a connection
    info = client.server_info()
    print(f"[OK] Success! Connected in {time.time() - start:.2f}s")
    print(f"Server version: {info.get('version')}")
except Exception as e:
    print(f"[FAILED] Failed: {e}")
