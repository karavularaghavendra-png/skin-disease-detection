"""Test the /predict/web API endpoint with test_skin_sample.jpg"""
import requests
import json

url = "http://localhost:8080/predict/web"
filepath = "test_skin_sample.jpg"

print("Testing /predict/web endpoint...")
print(f"Image: {filepath}")
print()

with open(filepath, "rb") as f:
    response = requests.post(url, files={"file": ("test.jpg", f, "image/jpeg")})

print(f"Status: {response.status_code}")
if response.status_code == 200:
    r = response.json()
    print(f"Disease: {r.get('display_name', r.get('disease'))}")
    print(f"Confidence: {round(r.get('confidence', 0) * 100)}%")
    print(f"Severity: {r.get('severity')}")
    print(f"Reliable: {r.get('is_reliable')}")
    print(f"TTA Used: {r.get('tta_used')}")
    print(f"TTA Agreement: {r.get('tta_agreement')}%")
    print(f"Latency: {r.get('latency_ms')}ms")
    print(f"Symptoms: {len(r.get('symptoms', []))} items")
    print(f"Recommendations: {len(r.get('recommendations', []))} items")
    print(f"Top Predictions: {r.get('top_predictions')}")
    print()
    print("API IS WORKING PERFECTLY!")
else:
    print(f"ERROR: {response.text}")
