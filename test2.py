import base64
import requests
from PIL import Image
import io

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"  # same as app

def identify_food(image_path):
    # Load image
    image = Image.open(image_path)

    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {
                "role": "system",
                "content": "You are a food recognition expert. Identify the dish name accurately."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify the dish in this image. Respond with ONLY the dish name."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )

    print("STATUS:", response.status_code)
    print("RAW RESPONSE:\n", response.json())

    try:
        answer = response.json()["choices"][0]["message"]["content"]
        print("\nMODEL OUTPUT:", answer)
    except Exception as e:
        print("Parsing error:", e)

if __name__ == "__main__":
    identify_food("brave_screenshot_search.brave.com (2).png")  # <-- put any food image here
