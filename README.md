# CLIP text-to-image

## Overview
This repository explores the concept of using OpenAI's CLIP model to guide itself towards generating an image from Gaussian noise based on a text prompt.

Run `clipga-self-text-to-image.py` for a gradient ascent process that optimizes an image to match a given text prompt (defined at the very end!). The script also saves `.npy` files, which you can analyze with `clipga-numpy-analysis.py`.

## Prerequisites
- This project requires [OpenAI's CLIP](https://github.com/openai/CLIP).

## How It Works
Typically, CLIP is used to guide text-to-image AI systems like Stable Diffusion towards your text prompt, by evaluating the alignment between generated images and textual descriptions. In this unique experiment, CLIP attempts to direct its own image generation process.

## Why?!
CLIP's neurons are highly multimodal, making it challenging for the model to converge on a single coherent image. This project serves more as an exploration into eXplainable AI (XAI), offering insight into CLIP's "perception," rather than functioning as a typical generative AI model.

By default as defined in the script, CLIP optimizes towards "❤️". And that's not just a heart emoji to CLIP - it is the text "love" and "I love you", it is mouths, it is holding hands, it is wedding rings. You get the idea. Many related concepts.
It doesn't converge to anything, it keeps oscillating around any related concepts, but saves the images for you to gaze at.

![4image_260](https://github.com/zer0int/CLIP-text-to-image/assets/132047210/fc01888f-ace1-4563-bfcd-20194a7ac9c3) ![2image_160](https://github.com/zer0int/CLIP-text-to-image/assets/132047210/bc7eac36-dd08-4896-9e6b-987c54a8ca4b) ![6image_510](https://github.com/zer0int/CLIP-text-to-image/assets/132047210/4f3101c9-57b3-4d2a-abba-fb019eac510c) ![9image_980](https://github.com/zer0int/CLIP-text-to-image/assets/132047210/3ac6cdc4-a6f5-4040-9d24-122519b5db5d)

## Acknowledgements
A significant portion of the coding for this project was performed by GPT-4 from OpenAI. Their collaboration made this exploration into the capabilities of CLIP possible.


Also, GPT-4 wrote all of the above except the paragraph starting with "By default [...]" and this one. Thanks, AI. <3

PS: If the custom loss function is weird, please consult GPT-4; it came up with that. Seriously.

