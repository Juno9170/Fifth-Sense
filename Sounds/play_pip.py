import sounddevice as sd
import numpy as np
import wave
import math

def play_stereo_panned(audio_file_path, angle):
    """Plays a stereo audio file panned to a specific angle.

    Args:
        audio_file_path (str): Path to the stereo audio file.
        angle (float): Panning angle in degrees (-90 to +90, where -90 is full left,
                       0 is center, and +90 is full right).
    """
    try:
        # Open the WAV file
        with wave.open(audio_file_path, 'rb') as wf:
            # Extract audio properties
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()

            # Read all frames from the WAV file
            raw_data = wf.readframes(num_frames)

        # Convert byte data to integer samples (NumPy array)
        if sample_width == 1:
            dtype = np.int8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width")

        audio_data = np.frombuffer(raw_data, dtype=dtype)

        # Reshape to stereo
        audio_data = audio_data.reshape(-1, num_channels)

        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32) / np.iinfo(dtype).max

        # Calculate panning gains
        angle_rad = angle * (math.pi / 180)
        left_gain = math.cos(angle_rad)
        right_gain = math.sin(angle_rad)

        # Apply panning
        audio_data[:, 0] *= left_gain   # Left channel
        audio_data[:, 1] *= right_gain  # Right channel

        # Play the audio using sounddevice
        sd.play(audio_data, samplerate=frame_rate)
        sd.wait()  # Wait until playback is finished

        print(f"Stereo audio played panned at {angle} degrees.")

    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage:
audio_file = "440.wav"  # Replace with your audio file path
panning_angle = 45  # Ahead and to the left
play_stereo_panned(audio_file, panning_angle)
