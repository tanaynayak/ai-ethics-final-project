import os
import pandas as pd
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from tqdm import tqdm

# Initialize the Koala Lightning 1B pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "etri-vilab/koala-lightning-1b",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Configure the scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

# Load the CSV file
csv_file = "prompts.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Create the output directory
output_dir = "koala_lightning_1b"
os.makedirs(output_dir, exist_ok=True)

# Generate images with tqdm progress bar
for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing Prompts"):
    prompt = row["Prompt"]
    safe_prompt = prompt.replace(" ", "_").replace(",", "").replace(".", "").lower()
    negative_prompt = "worst quality, low quality, illustration, low resolution"

    for i in range(1, 6):  # Generate 5 images for each prompt
        output_path = os.path.join(output_dir, f"{safe_prompt}_{i}.png")
        
        # Generate the image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=3.5,  # Adjust scale as needed
            num_inference_steps=10  # Adjust number of steps if required
        ).images[0]
        
        # Save the image
        image.save(output_path)

print(f"All images saved in '{output_dir}' directory.")