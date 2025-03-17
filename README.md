# Sixth-Sense: A Symphony of Sound and Speech for Enhanced Spatial Awareness

Imagine navigating the world with a richer understanding of your surroundings, not just through sight, but through a symphony of sound and descriptive language. That's the promise of Sixth-Sense, a project designed to provide real-time 3D awareness to visually impaired individuals using cutting-edge AI and spatial audio technology.

## Here's how we're transforming perception:

- 🤖 **AI-Powered Perception**: At the heart of Sixth-Sense lies a powerful AI core. We've harnessed the speed and accuracy of YOLOv11 to identify objects in the environment with remarkable precision. Complementing this, DepthAnythingV2 estimates the distance to these objects, creating a detailed 3D map of the surroundings. Combined, the two provide a precise spatial description of the user's environment.

- 🧠 **Scene Understanding with Gemini**: But it's not just about detecting objects; it's about understanding the scene. Google's Gemini model analyzes the identified objects and their spatial relationships, generating rich, descriptive narratives that paint a picture of the environment.

- 🎧 **Spatial Audio Immersion**: To translate this 3D information into an intuitive experience, we use Head-Related Transfer Functions (HRTF) to create directional beeps. These beeps provide a spatial audio landscape, allowing the user to pinpoint the location of objects with remarkable accuracy. The closer the object, the louder the beep, and the direction of the beep matches its position in the real world.

- 🗣️ **Real-Time Narration**: Finally, our system uses advanced Text-to-Speech (TTS) to deliver real-time verbal descriptions of the scene, reinforcing the spatial audio cues and providing a comprehensive understanding of the environment.

## How We Built It:

Our project leverages a distributed architecture. A laptop handles the computationally intensive tasks of object detection, depth estimation, and TTS. A Raspberry Pi running QNX is used for modal control, enabling the use of Gemini to describe scenes, toggling spatial audio cues, and turning the device off. The laptop is then responsible for determining where the closest objects to the user are and playing the spatial audio cues.

## Challenges and Triumphs:

- 🤯 **Libraries with No Clue**: We wanted spatial audio, so we started with basic panning and volume tricks. Didn't cut it. We dug into 3D Audio HRTF Processing to make sounds feel real, but the libraries were a mess – zero docs, and no example code anyone could use. Even ChatGPT was stumped! We finally figured it out by just grinding through it.

- 🤕 **More Documentation Headaches**: Adding speech with Google's Text-to-Speech was rough. Their documentation felt like it was written for experts. We spent ages fighting with authentication, API stuff, and figuring out how to make it work. Lots of trial and error (and Googling) later, we finally got it.

- 🧵 **We Need More Threads!**: When trying to play both spatial audio cues and Gemini's text-to-speech descriptions in the same loop, we came across a _lot_ of issues with segmentation faults and threading errors. Having never worked with multithreading before, we spent a significant amount of time trying to resolve this issue, mostly to no avail. Finally, we were able to architect a solution involving two separate audio queues and a mutual exclusion lock.

## What's Next:

- 🌍 Our vision for Sixth-Sense extends beyond the current prototype. We plan to expand TTS support to multiple languages, making the technology accessible to a global audience.

- 📏 We realized very quickly that while DepthAnythingV2 is an incredible model and certainly has many super cool uses, the field of Monocular Depth Estimation, it doesn't currently excel in precision in its measurements. To take Sixth-Sense to the next level, we are looking to LIDAR as an alternative solution.

## Team

- [Ryan Zhu](https://github.com/Juno9170) & [Patrick Yuan](https://github.com/holycactusjuice) - AI/ML Engineers
- [Uros Petrovic](https://github.com/crooder1) & [Puneet Singh](https://github.com/punz1738) - Embedded Systems/Hardware
