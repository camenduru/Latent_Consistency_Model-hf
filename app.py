#!/usr/bin/env python
from __future__ import annotations

import os
import random
import time

import gradio as gr
import numpy as np
import PIL.Image
import torch

from diffusers import DiffusionPipeline
import torch

import os
import torch
from tqdm import tqdm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from concurrent.futures import ThreadPoolExecutor
import uuid
import cv2

DESCRIPTION = '''# Latent Consistency Model
Distilled from [Dreamshaper v7](https://huggingface.co/Lykon/dreamshaper-7) fine-tune of [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) with only 4,000 training iterations (~32 A100 GPU Hours). [Project page](https://latent-consistency-models.github.io)
'''
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"
DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse

pipe = DiffusionPipeline.from_pretrained("ckpt/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main", safety_checker=None)
pipe.to(torch_device="cuda", torch_dtype=DTYPE)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def save_image(img):
    unique_name = str(uuid.uuid4()) + '.png'
    img.save(unique_name)
    return unique_name

def save_images(image_array):
    paths = []
    with ThreadPoolExecutor() as executor:
        paths = list(executor.map(save_image, image_array))
    return paths

def generate(
    prompt: str,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    progress = gr.Progress(track_tqdm=True)
) -> PIL.Image.Image:
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    start_time = time.time()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        lcm_origin_steps=50,
        output_type="pil",
    ).images
    paths = save_images(result)
    print(time.time() - start_time)
    return paths, seed

examples = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery", grid=[2]
        )
    with gr.Accordion("Advanced options", open=False):
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
            randomize=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed across runs", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=512,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=512,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance scale for base",
                minimum=2,
                maximum=14,
                step=0.1,
                value=8.0,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps for base",
                minimum=1,
                maximum=8,
                step=1,
                value=4,
            )
        with gr.Row():
            num_images = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=8,
                step=1,
                value=4,
                visible=False,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            num_images,
            randomize_seed
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(api_open=False)
    # demo.queue(max_size=20).launch()
    demo.launch(share=True)
