from app.main_advanced import USERS, _save_local_users, _USERS_FILE
import os

def verify_local_persistence():
    test_user = "verified_user"
    USERS[test_user] = {
        'password': 'password123',
        'role': 'Doctor',
        'name': 'Dr. Verified',
        'access_level': 'admin'
    }
    
    print(f"Saving '{test_user}' to local cache...")
    _save_local_users()
    
    if _USERS_FILE.exists():
        size = os.path.getsize(_USERS_FILE)
        print(f"[OK] Success! Local users file created ({size} bytes)")
        with open(_USERS_FILE, 'r') as f:
            print(f"Content preview: {f.read()[:100]}...")
    else:
        print("[FAILED] Error: Local users file was NOT created!")

if __name__ == "__main__":
    verify_local_persistence()
