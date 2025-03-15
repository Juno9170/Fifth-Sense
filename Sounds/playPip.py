import openal
import numpy as np


def play_sound_at_position(sound_file, position):
    """
    Plays a short sound file at a specified 3D position using PyOpenAL.
    """
    try:
        # Load the sound file into a Source object
        source = openal.oalOpen(sound_file)

        if not source:
            raise Exception(f"Failed to load sound file: {sound_file}")

        source.set_source_relative(True)
        # Set the source position
        source.set_position(position)

        # Play the sound
        source.play()

        # Wait for the sound to finish playing (check source state)
        while source.get_state() == openal.AL_PLAYING:
            pass

        # Clean up: stop and destroy the source.
        source.stop()
        del source

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        # Initialize OpenAL
        if not openal.oalGetInit():
            openal.oalInit()

        sound_file = "440.wav"
        position = (100, 5, -20)

        play_sound_at_position(sound_file, position)

    finally:
        # Ensure OpenAL is closed even if errors occur
        if openal.oalGetInit():
            openal.oalQuit()
