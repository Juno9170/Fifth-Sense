# [Sixth-Sense]([url](https://devpost.com/software/sixth-sense-eczxnv))

Bringing back not only the fifth, but also a sixth sense for the visually impaired. Powered by the bleeding edge in artificial intelligence and embedded systems, we aim to revolutionize accessibility.

## Overview

Sixth Sense is an innovative assistive technology designed to assist visually impaired individuals by providing real-time 3D object detection and scene understanding. It leverages AI models for depth estimation, object detection, and scene description to create an intuitive and informative experience.

The system leverages the processing power of an M4 MacBook to handle camera input and complex AI processing, while a Raspberry Pi with QNX acts as a remote controller, managing communication and feedback through high-speed socket transmission. The system delivers real-time spatial audio feedback, allowing users to perceive the environment through sound.

## Features

- 3D Object Detection:
    - Uses [YOLOv11 (Ultralytics)](https://github.com/ultralytics/ultralytics) for real-time object detection.

- Depth Estimation:
    - Employs [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth estimation to map objects in a 3D space.

- Scene Description:
    - Utilizes [Google Gemini](https://deepmind.google/technologies/gemini/) to generate contextual scene descriptions.

- Spatial Audio with HRTF:
    - Uses HRTF (Head-Related Transfer Function) to simulate 3D audio positioning.

- Text-to-Speech (TTS):
    - Converts scene descriptions into natural speech using [Google Cloud TTS](https://cloud.google.com/text-to-speech).
    - Provides real-time audio descriptions of the environment.

- Remote Control:
    - Raspberry Pi with [QNX](https://blackberry.qnx.com/en/products/foundation-software/qnx-software-development-platform) sends control signals to the MacBook using socket communication.
    - MacBook handles real-time camera input and processing.

## How It Works

1. **MacBook captures camera input** – The MacBook’s camera captures a live video feed.
2. **YOLOv11 identifies objects** – YOLOv11 detects and classifies objects.
3. **DepthAnythingV2 estimates depth** – Generates a 3D depth map of the scene.
4. **Google Gemini provides scene descriptions** – Generates a detailed, natural language summary of the scene.
5. **Spatial audio feedback with HRTF** – A directional beep is played based on the position of the nearest objects.
6. **Text-to-Speech (TTS)** – Scene descriptions are converted to speech and played through headphones.

## Team

- [Ryan Zhu](https://github.com/Juno9170) & [Patrick Yuan](https://github.com/holycactusjuice) - AI/ML Engineers
- [Uros Petrovic](https://github.com/crooder1) & [Puneet Singh](https://github.com/punz1738) - Embedded Systems/Hardware

## Experience the power of perception with Sixth Sense!
