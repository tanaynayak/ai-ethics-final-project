import os
import pandas as pd
import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm

# Initialize the Stable Diffusion pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16, 
    token="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"
)
pipe = pipe.to("cuda")

# Load the CSV file
csv_file = "prompts.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Create the output directory
output_dir = "stabilityai_stable_diffusion_3"
os.makedirs(output_dir, exist_ok=True)

# Generate images with tqdm progress bar
for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing Prompts"):
    prompt = row["Prompt"]
    safe_prompt = prompt.replace(" ", "_").replace(",", "").replace(".", "").lower()
    
    for i in range(1, 6):  # Generate 5 images for each prompt
        output_path = os.path.join(output_dir, f"{safe_prompt}_{i}.png")
        
        # Generate the image
        image = pipe(
            prompt,
            negative_prompt="",  # Add any negative prompt if needed
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images[0]
        
        # Save the image
        image.save(output_path)

print(f"All images saved in '{output_dir}' directory.")