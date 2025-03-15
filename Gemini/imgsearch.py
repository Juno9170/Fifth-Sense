import os
import requests
import PIL.Image
import numpy as np
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

def analyze_image_with_gemini(image_data: np.ndarray, prompt: str) -> str:
    """Analyzes an image using Google Gemini API and returns the response text.
    
    Args:
        image_data (np.ndarray): Raw image data as a NumPy array.
        prompt (str): The text prompt for the AI.
    
    Returns:
        str: The response text from the API.
    """
    # Load API key
    GEMINI_KEY = os.getenv('GEMINI_KEY')
    if not GEMINI_KEY:
        raise ValueError("GEMINI_KEY is missing. Please set it in the environment variables.")

    try:
        # Convert numpy ndarray to PIL Image
        image = PIL.Image.fromarray(image_data)

        print("Done tsts")

        # Initialize Gemini client
        client = genai.Client(api_key=GEMINI_KEY)

        # Send request to the model
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt, image]
        )

        return response.text  # Return response instead of printing

    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

# Example Usage
if __name__ == "__main__":
    # Example numpy ndarray representing an image
    image_data = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # Example random image

    prompt = "Can you describe what it feels like to be in this image? Speak like you are currently talking to a blind friend next to you, dont use Imagine the-. Describe briefly where things are located be as consice and objective as possible dont make a list just a couple sentences."
    
    try:
        result = analyze_image_with_gemini(image_data, prompt)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
