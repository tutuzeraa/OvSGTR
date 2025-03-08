from PIL import Image
import os
import json

custom_dir = "custom_data"
output_file = "custom_data/custom_data.json"

image_data = []
start_id = 1000
for idx, filename in enumerate(os.listdir(custom_dir)):
    if filename.endswith(".jpg"):
        img_path = os.path.join(custom_dir, filename)
        with Image.open(img_path) as img:
            width, height = img.size
        image_id = start_id + idx
        image_data.append({
            "image_id": image_id,
            "width": width,
            "height": height,
            "file_name": filename
        })

with open(output_file, "w") as f:
    json.dump(image_data, f, indent=2)
print(f"Saved metadata to {output_file}")