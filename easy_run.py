#!/usr/bin/env python3
# import pipeline and scheduler from https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7/
from lcm_pipeline import LatentConsistencyModelPipeline
from lcm_scheduler import LCMScheduler
import hf_image_uploader as hiu
import torch

scheduler = LCMScheduler.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")

pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", scheduler=scheduler)
pipe.to("cuda", dtype=torch.float16)

prompt = "a red horse"
images = pipe(prompt=prompt, guidance_scale=8.0, num_inference_steps=4, lcm_origin_steps=50, output_type="pil").images

for image in images:
    hiu.upload(image, "patrickvonplaten/images")
