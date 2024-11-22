import requests
from config import BASE_URL

def process_image(image, api_key):
    url = BASE_URL + "process_result"
    files = {"image": image}
    headers = {"x-api-key": api_key}
    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        return response.json().get("result_url")
    except requests.exceptions.RequestException as e:
        print(f"Image processing error: {e}")
        return None
