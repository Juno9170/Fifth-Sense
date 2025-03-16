import subprocess
import requests
import json
import base64
import io
import wave
import numpy as np

def synthesize_speech(text, project_id="gen-lang-client-0500175658"):
    """Synthesizes speech from the given text using Google Cloud Text-to-Speech API."""
    try:
        # Replace with the actual path to gcloud.exe
        gcloud_path = "/Users/pat/Downloads/google-cloud-sdk/bin/gcloud"
        access_token = subprocess.check_output(
            [gcloud_path, "auth", "print-access-token"]).decode("utf-8").strip()

        # Construct headers
        headers = {
            "Content-Type": "application/json",
            "X-Goog-User-Project": project_id,
            "Authorization": f"Bearer {access_token}"
        }

        # Construct request body
        body = {
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": "en-US-Chirp3-HD-Aoede"
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "speakingRate": 1.0
            }
        }

        # Send request to Text-to-Speech API
        response = requests.post(
            "https://texttospeech.googleapis.com/v1beta1/text:synthesize",
            headers=headers,
            json=body
        )
        response.raise_for_status()  # Raise HTTPError for bad responses

        return response.json()

    except subprocess.CalledProcessError as e:
        print(f"Error getting access token: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None


# Example usage
text_to_synthesize = "Sixth sense. starting!"
output_filename = "output.wav"
speech_response = synthesize_speech(text_to_synthesize)

if speech_response:
    audio_content = speech_response.get("audioContent")
    if audio_content:
        # Decode the base64-encoded audio content
        audio_data = base64.b64decode(audio_content)

        # Save directly to a WAV file
        with wave.open(output_filename, 'wb') as wf:
            with wave.open(io.BytesIO(audio_data), 'rb') as source_wf:
                # Copy parameters from source
                wf.setnchannels(source_wf.getnchannels())
                wf.setsampwidth(source_wf.getsampwidth())
                wf.setframerate(source_wf.getframerate())
                # Write the audio data
                wf.writeframes(source_wf.readframes(source_wf.getnframes()))
        
        print(f"Audio saved to {output_filename}")
    else:
        print("No audio content found.")
else:
    print("Speech synthesis failed.")
