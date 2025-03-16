import subprocess
import requests
import json
import base64
import io
import wave
import numpy as np
import sounddevice as sd


def synthesize_speech(text, project_id="gen-lang-client-0500175658"):
    """Synthesizes speech from the given text using Google Cloud Text-to-Speech API."""
    try:
        # Replace with the actual path to gcloud.exe
        gcloud_path = r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
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
text_to_synthesize = "Movies, oh my gosh, I just just absolutely love them. They're like time machines taking you to different worlds and landscapes, and um, and I just can't get enough of it."
speech_response = synthesize_speech(text_to_synthesize)

if speech_response:

    audio_content = speech_response.get("audioContent")
    if audio_content:
        # Decode the base64-encoded audio content
        audio_data = base64.b64decode(audio_content)

        # Use wave to read the audio data from the decoded bytes
        with wave.open(io.BytesIO(audio_data), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            # Convert audio frames to a numpy array of type int16
            audio_array = np.frombuffer(frames, dtype=np.int16)
            # Reshape the array if the audio has multiple channels
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)

        # Play the audio using sounddevice (no extra window opens)
        sd.play(audio_array, sample_rate)
        sd.wait()  # Wait until audio is finished playing
    else:
        print("No audio content found.")
else:
    print("Speech synthesis failed.")
