import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add app to path
sys.path.append(str(Path(__file__).parent))

load_dotenv()

from app.mongodb_database import get_database, users_col

def test_registration():
    username = "testuser_persistence"
    new_user = {
        'username': username,
        'password': 'password123',
        'role': 'Doctor',
        'name': 'Dr. Test',
        'access_level': 'admin',
    }
    
    print(f"Attempting to persist user '{username}' to Atlas...")
    try:
        # 1. Access collection
        col = users_col()
        print("[OK] Obtained users collection accessor")
        
        # 2. Try insert/upsert
        result = col.update_one(
            {'username': username},
            {'$set': new_user},
            upsert=True
        )
        print(f"[OK] Upsert successful! Matched: {result.matched_count}, Upserted ID: {result.upserted_id}")
        
        # 3. Verify read back
        doc = col.find_one({'username': username})
        if doc:
            print(f"[OK] Verified: Found user in Atlas: {doc['name']}")
        else:
            print("[FAILED] Error: User not found in Atlas after upsert!")
            
    except Exception as e:
        print(f"[FAILED] Persistence failed: {e}")

if __name__ == "__main__":
    test_registration()
