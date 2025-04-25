from pathlib import Path
import tqdm
import torch 
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import  pipeline , set_seed
import matplotlib.pyplot as plt
import cv2
 
class CFG:
    device = "cpu"
    seed = 42
    generator = torch.Generator("cpu").manual_seed(seed)
    image_gen_step = 3
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (500 , 500)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model = StableDiffusionPipeline.from_pretrained(CFG.image_gen_model_id,torch_dtype = torch.get_float32_matmul_precision , use_auth_token = "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" ,guidance_scale = 9)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt , model ):
    image = model(
     prompt , num_inference_steps = CFG.image_gen_step,
     generator = CFG.generator,
     guidance_scale = CFG.image_gen_guidance_scale   
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image 
imge = generate_image("two trains" , image_gen_model)
plt.imshow(imge)
plt.axis("off")
plt.title("Generated Image")
plt.show()

