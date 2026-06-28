import requests

url = 'https://api.xah.io/v1/chat/completions'
headers = {
    'Authorization': 'Bearer sk-b587319f2e69e3f07e219eea70b3d5ce83dd6a20a2e0a95039c4b84557e2c010',
    'Content-Type': 'application/json',
}
payload = {
    "model": "deepseek-v4-flash",
    "messages": [
        {
            "role": "user",
            "content": "Xin chào, bạn là ai?"
        }
    ]
}

res = requests.post(url, headers=headers, json=payload, timeout=120)
print(res.status_code, res.json())