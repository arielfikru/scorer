import json
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
import argparse

# Define default values for certain variables
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_SIZE = 3

# Create the argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for image classification")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for image classification")
parser.add_argument("--img_dir", type=str, help="Path to the directory containing images")
parser.add_argument("--output_dir", type=str, help="Path to the output JSON file")
parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Maximum number of worker threads to use in the thread pool")
parser.add_argument("--recursive", action="store_true", help="Include full path of the image in the output JSON")

# Parse the arguments
args = parser.parse_args()

# Get the values of the arguments
batch_size = args.batch_size
use_gpu = args.use_gpu
parent_img_dir = args.img_dir
output_dir = args.output_dir
max_workers = args.max_workers
recursive = args.recursive

# Create a list to store the results for each image
results = []

# Create a cache to store the results of the classification tasks
cache = {}

# Set device for pipeline
device = 0 if use_gpu else -1

# Initialize pipelines outside of the function
pipe_aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic", device=device, batch_size=batch_size)
pipe_style = pipeline("image-classification", "cafeai/cafe_style", device=device, batch_size=batch_size)
pipe_waifu = pipeline("image-classification", "cafeai/cafe_waifu", device=device, batch_size=batch_size)
pipe_nsfw  = pipeline("image-classification", "carbon225/vit-base-patch16-224-hentai", device=device, batch_size=batch_size)

# Function to process images in a directory
def process_images(img_dir):
    num_images = len([f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")])

    for file in tqdm(os.listdir(img_dir), total=num_images, dynamic_ncols=True):
        if file.endswith(".jpg") or file.endswith(".png"):
            filepath = os.path.join(img_dir, file)
            input_img = Image.open(filepath)
            
            # Determine the filename for the JSON based on the recursive argument
            json_filename = filepath if recursive else file

            # Use the aesthetic classifier
            data = pipe_aesthetic(input_img, top_k=2)
            final = {}
            for d in data:
                final[d["label"]] = d["score"]

            # Use the style classifier
            data = pipe_style(input_img, top_k=5)
            final_style = {}
            for d in data:
                final_style[d["label"]] = d["score"]

            # Use the waifu classifier
            data = pipe_waifu(input_img, top_k=5)
            final_waifu = {}
            for d in data:
                final_waifu[d["label"]] = d["score"]
             
            # Use the nsfw classifier
            data = pipe_nsfw(input_img, top_k=3)
            final_nsfw = {}
            for d in data:
                final_nsfw[d["label"]] = d["score"]

            # Check if the results for this image are in the cache
            if json_filename in cache:  # <-- change this line
                # Use the cached results
                result = cache[json_filename]  # <-- change this line
            else:
                # Store the results for this image in the cache
                result = {"filename": json_filename, "aesthetic": final, "style": final_style, "waifu": final_waifu, "nsfw": final_nsfw}
                cache[json_filename] = result  # <-- change this line

            # Submit the task to the thread pool
            future = executor.submit(results.append, result)
            
# Create a thread pool with the specified number of worker threads
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Use os.walk to recursively walk through the parent directory and all its subdirectories
    for dirpath, dirnames, filenames in os.walk(parent_img_dir):
        process_images(dirpath)
    # Save the results to a JSON file
    with open(output_dir, "w") as f:
      json.dump(results, f, indent=2)
