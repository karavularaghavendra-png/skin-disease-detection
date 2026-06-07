"""Quick API smoke-test — sends a real skin image to /predict/web and prints the result."""
import urllib.request
import json
import os

img_path = os.path.join("dataset", "skin_dataset", "test", "acne", "acne_0000.jpg")
if not os.path.exists(img_path):
    print(f"Test image not found: {img_path}")
    exit(1)

with open(img_path, "rb") as f:
    img_data = f.read()

boundary = "----PythonTestBoundary"
body = b""
body += ("--" + boundary + "\r\n").encode()
body += b'Content-Disposition: form-data; name="file"; filename="acne_0000.jpg"\r\n'
body += b"Content-Type: image/jpeg\r\n\r\n"
body += img_data
body += ("\r\n--" + boundary + "--\r\n").encode()

req = urllib.request.Request(
    "http://localhost:8000/predict/web",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

try:
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    print("=" * 50)
    print("  PREDICTION API TEST RESULT")
    print("=" * 50)
    print(f"  STATUS:         {resp.status}")
    print(f"  DISEASE:        {result.get('disease')}")
    print(f"  DISPLAY NAME:   {result.get('display_name')}")
    print(f"  CONFIDENCE:     {result.get('confidence')}")
    print(f"  SEVERITY:       {result.get('severity')}")
    print(f"  TTA USED:       {result.get('tta_used')}")
    print(f"  TTA PASSES:     {result.get('tta_passes')}")
    print(f"  TTA AGREEMENT:  {result.get('tta_agreement')}")
    print(f"  IS RELIABLE:    {result.get('is_reliable')}")
    print(f"  LATENCY:        {result.get('latency_ms')}ms")
    print(f"  SYMPTOMS:       {result.get('symptoms', [])[:3]}")
    print(f"  RECOMMENDATIONS:{result.get('recommendations', [])[:2]}")
    print(f"  TOP PREDICTIONS:{result.get('top_predictions', [])}")
    print("=" * 50)
    print("  PREDICTION API TEST PASSED!")
    print("=" * 50)
except urllib.error.HTTPError as e:
    body_text = e.read().decode()
    print(f"ERROR {e.code}: {body_text}")
except Exception as e:
    print(f"EXCEPTION: {e}")
