import requests
import random
import json

print("Constructing simulated search payload...")

# Simulate 3 different web pages (documents) competing for the top spot
documents = []
for i in range(1, 4):
    documents.append({
        "doc_id": f"url_candidate_00{i}",
        # Generate 136 random floats to simulate MSLR-WEB10K features (BM25, PageRank, etc.)
        "features": [random.random() for _ in range(136)]
    })

payload = {
    "query_id": "google_search_query_8472",
    "documents": documents,
    "top_k": 3
}

print("Firing payload at LambdaMART microservice...\n")

# Change this line:
response = requests.post("http://127.0.0.1:8001/rank", json=payload)

if response.status_code == 200:
    print("✅ SUCCESS: Ranked Results")
    print("-" * 30)
    print(json.dumps(response.json(), indent=2))
else:
    print(f"❌ ERROR: {response.status_code}")
    print(response.text)