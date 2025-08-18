import os
from dotenv import load_dotenv

# Load .env into os.environ
load_dotenv()

API_KEY = os.getenv("VELLUM_API_KEY")
if not API_KEY:
    raise RuntimeError("VELLUM_API_KEY not found in environment variables.")

import requests

url = "https://api.vellum.ai/v1/deployments/provider-payload"
headers = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json",
}
body = {
    "deployment_name": "plan-name-identification-prompt-v-18-0-variant-1",
    "inputs": [
        {"type": "STRING", "name": "your_input_var", "value": "example"}
    ]
}
r = requests.post(url, json=body, headers=headers)
print(r.status_code, r.text["text"])
