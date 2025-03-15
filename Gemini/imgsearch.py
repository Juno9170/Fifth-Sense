
import requests
from google.genai import types
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_KEY = os.getenv('GEMINI_KEY')


image_path = "https://goo.gle/instrument-img"
image = requests.get(image_path)

client = genai.Client(api_key=GEMINI_KEY)
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What is this image? Can you describe what it would feel like to touch be concise.",
              types.Part.from_bytes(data=image.content, mime_type="image/jpeg")])

print(response.text)
