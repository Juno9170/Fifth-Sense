import subprocess
import requests
import json
import base64
import io
import wave
import numpy as np
import sounddevice as sd
import threading
from queue import Queue
import time
import librosa

# Add a global audio lock that both systems will use
audio_lock = threading.Lock()

class TTSPlayer:
    def __init__(self):
        self.audio_queue = Queue()
        self.running = True
        self.thread = None
        self.stream = None

    def start(self):
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _process_queue(self):
        while self.running:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                self._play_audio(audio_data)
            else:
                time.sleep(0.1)

    def _play_audio(self, audio_data):
        try:
            # Acquire the global audio lock before playing
            with audio_lock:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None

                with wave.open(io.BytesIO(audio_data), "rb") as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    if channels > 1:
                        audio_array = audio_array.reshape(-1, channels)
                    
                    self.stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=channels,
                        dtype=np.int16
                    )
                    self.stream.start()
                    self.stream.write(audio_array)
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None

        except Exception as e:
            print(f"Error playing audio: {e}")
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                self.stream = None

# Create a global TTS player instance
tts_player = TTSPlayer()
tts_player.start()

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
                "speakingRate": 1.3
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


def speak(text_to_synthesize):
    speech_response = synthesize_speech(text_to_synthesize)

    if speech_response:
        audio_content = speech_response.get("audioContent")
        if audio_content:
            # Decode the base64-encoded audio content
            audio_data = base64.b64decode(audio_content)
            # Add to queue instead of playing directly
            tts_player.audio_queue.put(audio_data)
        else:
            print("No audio content found.")
    else:
        print("Speech synthesis failed.")


if __name__ == "__main__":
    try:
        text_to_synthesize = "Hey, so I'm sitting on a comfy couch, I can feel the texture of it right here behind my back, its a nice dark gray fabric. And in front of me, someone's holding up a big board full of stickers, I can feel the cool wall behind me."
        speak(text_to_synthesize)
    finally:
        tts_player.stop()
