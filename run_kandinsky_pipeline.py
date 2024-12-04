import os
import pandas as pd
import torch
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm

# Initialize the Kandinsky pipeline
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Load the CSV file
csv_file = "prompts.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Create the output directory
output_dir = "kandinsky_2_2"
os.makedirs(output_dir, exist_ok=True)

# Generate images with tqdm progress bar
for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing Prompts"):
    prompt = row["Prompt"]
    safe_prompt = prompt.replace(" ", "_").replace(",", "").replace(".", "").lower()
    negative_prompt = "low quality, bad quality"  # Customize negative prompt as needed

    for i in range(1, 6):  # Generate 5 images for each prompt
        output_path = os.path.join(output_dir, f"{safe_prompt}_{i}.png")
        
        # Generate the image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prior_guidance_scale=1.0,  # Adjust guidance scale if needed
            height=768, 
            width=768
        ).images[0]
        
        # Save the image
        image.save(output_path)

print(f"All images saved in '{output_dir}' directory.")