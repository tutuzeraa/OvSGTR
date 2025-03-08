import json
import numpy as np

custom_image_ids = [3000000]

image_data_path = '../data/visual_genome/stanford_filtered/image_data.json'

with open(image_data_path, 'r') as f:
    image_data = json.load(f)

custom_eval_images = [img for img in image_data if img['image_id'] in custom_image_ids]

print(f"Found {len(custom_eval_images)} custom images.")

custom_split_path = '../data/visual_genome/stanford_filtered/custom_eval_split.json'
with open(custom_split_path, 'w') as f:
    json.dump(custom_eval_images, f, indent=2)

print(f"Custom evaluation split saved to: {custom_split_path}")