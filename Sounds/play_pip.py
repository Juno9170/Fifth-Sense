import pygame
import math

def calculate_pan(angle_degrees):
    """
    Calculates the left/right pan values based on an angle.

    Args:
        angle_degrees: The angle in degrees, where 0 is center,
                       -90 is hard left, and 90 is hard right.

    Returns:
        A tuple (left_volume, right_volume) where each value is between 0.0 and 1.0.
    """

    # Normalize the angle to be between -1 and 1
    angle_normalized = angle_degrees / 90.0

    # Calculate left and right volumes.  A simple linear approach:
    right_volume = max(0, 1 - angle_normalized)  # Right volume increases as angle goes left
    left_volume = max(0, 1 + angle_normalized)   # Left volume increases as angle goes right

    # Ensure volumes are within the valid range (0.0 to 1.0)
    left_volume = min(1.0, left_volume)
    right_volume = min(1.0, right_volume)

    return (left_volume, right_volume)



# Initialize Pygame
pygame.init()

# --- Audio Setup ---
pygame.mixer.init()  # Initialize the mixer

# Load a sound file (replace 'your_sound.wav' with the actual path to your audio file)
try:
    sound = pygame.mixer.Sound('440.wav')
except pygame.error as e:
    print(f"Error loading sound: {e}")
    pygame.quit()
    exit()

# --- Screen Setup (for controlling the angle with the mouse) ---
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("3D Audio Demo")

# Create two channels for left and right
left_channel = pygame.mixer.Channel(0)
right_channel = pygame.mixer.Channel(1)

# Load sound into both channels
left_channel.play(sound)
right_channel.play(sound)

# Set volumes separately
left_channel.set_volume(0.1)
right_channel.set_volume(0.5)

# --- Game Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get mouse position and calculate angle
    mouse_x, mouse_y = pygame.mouse.get_pos()
    center_x = screen_width // 2
    angle_radians = math.atan2(mouse_y - screen_height // 2, mouse_x - center_x) # corrected height calculation for atan2
    angle_degrees = math.degrees(angle_radians)

    # Calculate pan values
    left_volume, right_volume = calculate_pan(angle_degrees)

    # Set volumes separately
    left_channel.set_volume(left_volume)
    right_channel.set_volume(right_volume)

    # Play the sound (if it's not already playing)
    if not pygame.mixer.get_busy():
        left_channel.play()
        right_channel.play()

    # Clear the screen and update the display (optional, just for visual feedback)
    screen.fill((0, 0, 0))  # Black background
    pygame.display.flip()

# Quit Pygame
pygame.quit()

