import requests

# The stored user credentials from db inspection
# Try to register/login fresh to get a valid token
BASE = "http://localhost:8081"

# Step 1: Register a test user
try:
    r = requests.post(f"{BASE}/auth/register", json={"email": "debugtest@test.com", "password": "test123456"})
    print("Register:", r.status_code, r.json())
    token = r.json().get("access_token")
except Exception as e:
    # Already exists, login
    r = requests.post(f"{BASE}/auth/login", json={"email": "debugtest@test.com", "password": "test123456"})
    print("Login:", r.status_code, r.json())
    token = r.json().get("access_token")

if not token:
    print("No token - cannot continue")
    exit(1)

headers = {"Authorization": f"Bearer {token}"}

# Step 2: Create a conversation
r = requests.post(f"{BASE}/conversations", json={"title": "Test Chat"}, headers=headers)
print("\nCreate Conv:", r.status_code)
print(r.json())

# Step 3: Get all conversations
r = requests.get(f"{BASE}/conversations", headers=headers)
print("\nGet Convs:", r.status_code)
convs = r.json()
print("Count:", len(convs))
if convs:
    print("First conv data:", convs[0])
