"""
One-time script: push all users from data/local_users.json to MongoDB Atlas.
Safe to re-run - uses upsert.
"""
import json
from dotenv import load_dotenv
load_dotenv()
from app.mongodb_database import users_col

with open('data/local_users.json') as f:
    users = json.load(f)

col = users_col()
for username, info in users.items():
    doc = {**info, 'username': username}
    col.update_one({'username': username}, {'$set': doc}, upsert=True)
    print(f'Upserted: {username}')

print(f'\nDone. Total users in Atlas: {col.count_documents({})}')
