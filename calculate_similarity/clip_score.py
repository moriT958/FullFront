"""
# CLIP Score Calculator
#
# This script calculates semantic similarity between pairs of images using OpenAI's CLIP model.
#
# Usage:
#   1. Install required packages: pip install torch clip-openai opencv-python pillow tqdm
#   2. Set the source folders in the main section:
#      - source_folder1: directory containing model-generated images
#      - source_folder2: directory containing reference/ground truth images
#   3. Set the output directory where results will be saved
#   4. Run the script: python clip_score.py
#
# The script will process matching image files in both directories and output JSON files
# with similarity scores for each model.
"""

import json

# pip install torch clip-openai opencv-python pillow tqdm
import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

import clip

# Disable PIL's DecompressionBombWarning
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("clip/ViT-B-32.pt", device=device)


def safe_open_image(image_path):
    """
    Safely open an image, compressing it if pixel count exceeds limit

    Args:
        image_path (str): Path to the image file

    Returns:
        PIL.Image: Processed image object
    """
    try:
        # First try to open the image
        img = Image.open(image_path)

        # Calculate total pixel count
        width, height = img.size
        pixels = width * height

        # If pixel count exceeds limit, compress the image
        # Use a smaller limit than PIL's warning threshold
        max_pixels = 89000000  # Slightly less than PIL warning threshold 89478485

        if pixels > max_pixels:
            # Calculate scale ratio
            scale = (max_pixels / pixels) ** 0.5

            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(
                f"Image {os.path.basename(image_path)} has been resized: {width}x{height} -> {new_width}x{new_height}"
            )

        return img
    except Exception as e:
        print(f"Cannot open image {image_path}: {str(e)}")
        # Try using cv2 to load the image as a fallback
        try:
            print(f"Trying to load image {image_path} with OpenCV")
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise Exception("OpenCV cannot load the image")
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            # Convert to PIL image
            img = Image.fromarray(img_rgb)
            return img
        except Exception as cv_error:
            print(f"Failed to load image {image_path} with OpenCV: {str(cv_error)}")
            raise


def clip_similarity(image_path1, image_path2):
    """
    Calculate CLIP semantic similarity between two images

    Args:
        image_path1 (str or PIL.Image): Path to the first image or PIL image object
        image_path2 (str or PIL.Image): Path to the second image or PIL image object

    Returns:
        similarity (float): Cosine similarity between the two images
    """
    # Load and preprocess images
    if isinstance(image_path1, str) and isinstance(image_path2, str):
        img1 = safe_open_image(image_path1)
        img2 = safe_open_image(image_path2)
    else:
        img1 = image_path1
        img2 = image_path2

    # Preprocess images and transfer to device
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img2 = preprocess(img2).unsqueeze(0).to(device)

    # Extract features using CLIP
    with torch.no_grad():
        features1 = model.encode_image(img1)
        features2 = model.encode_image(img2)

    # Normalize features
    features1 = features1 / features1.norm(p=2, dim=-1, keepdim=True)
    features2 = features2 / features2.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(features1, features2)

    return similarity.item()


def load_or_create_json(json_path):
    """
    Load or create JSON file

    Args:
        json_path (str): Path to JSON file

    Returns:
        dict: Loaded data or empty dictionary
    """
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                print(f"JSON file {json_path} parsing error, creating new file")
                return {"results": []}
    else:
        return {"results": []}


def save_json(data, json_path):
    """
    Save JSON data to file

    Args:
        data (dict): Data to save
        json_path (str): Path to JSON file
    """
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {json_path}")


def update_category_summary(data, category):
    """
    Update or create category summary item

    Args:
        data (dict): Dictionary containing results
        category (str): Dataset category name

    Returns:
        bool: Whether a new summary item was created
    """
    category_results = [
        item
        for item in data["results"]
        if item.get("Category") == category and item.get("id") != f"{category}_summary"
    ]

    if not category_results:
        return False

    # Calculate average scores
    clip_scores = [
        item.get("clip_score", 0)
        for item in category_results
        if isinstance(item.get("clip_score", 0), (int, float))
    ]

    avg_clip = np.mean(clip_scores) if clip_scores else 0

    # Check if summary item exists
    summary_exists = False
    for item in data["results"]:
        if item.get("id") == f"{category}_summary":
            item["clip_score"] = float(avg_clip)
            summary_exists = True
            break

    # If it doesn't exist, create new summary item
    if not summary_exists:
        data["results"].append(
            {
                "id": f"{category}_summary",
                "Category": category,
                "clip_score": float(avg_clip),
            }
        )
        return True

    return False


def compare_model_datasets(source_folder1, source_folder2, output_dir="./results"):
    """
    Compare image similarity across multiple models and datasets

    Args:
        source_folder1 (str): Path to first source folder containing model folders
        source_folder2 (str): Path to second source folder containing dataset folders
        output_dir (str): Output directory for JSON files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Model list
    models = [
        # "claude",
        # "gemini",
        # "internvl2",
        # "internvl3",
        # "llava",
        # "o1",
        "openai",
        # "qwen",
        # "o4mini",
        # "pro",
    ]
    # Dataset list
    datasets = [
        # "Code Refinement",
        "Image_to_code",
        # "Interaction_Authoring",
        # "Text_to_code",
    ]

    # Loop through all models
    for model in models:
        model_folder = os.path.join(source_folder1, model)
        if not os.path.exists(model_folder):
            print(f"Model folder {model_folder} does not exist, skipping")
            continue

        json_path = os.path.join(output_dir, f"{model}_similarity.json")

        # Load or create JSON file
        data = load_or_create_json(json_path)

        # Get current maximum ID
        current_ids = [
            item.get("id", "")
            for item in data["results"]
            if not isinstance(item.get("id"), str)
            or not item.get("id", "").endswith("_summary")
        ]
        current_max_id = max([int(i) for i in current_ids if str(i).isdigit()] + [0])

        # Loop through datasets
        for dataset in datasets:
            dataset_folder1 = os.path.join(model_folder, dataset)
            dataset_folder2 = os.path.join(source_folder2, dataset)

            if not os.path.exists(dataset_folder1) or not os.path.exists(
                dataset_folder2
            ):
                print(
                    f"Dataset folder {dataset_folder1} or {dataset_folder2} does not exist, skipping"
                )
                continue

            # Get image filenames from both folders
            files1 = [
                f
                for f in os.listdir(dataset_folder1)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]
            files2 = [
                f
                for f in os.listdir(dataset_folder2)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]

            # Find common filenames between the two folders
            common_files = list(set(files1) & set(files2))

            if not common_files:
                print(
                    f"Warning: No matching image files found in dataset {dataset} folders"
                )
                continue

            processing_count = 0

            # Calculate similarity for each pair of images
            for filename in tqdm(
                common_files, desc=f"Processing image similarity for {model}/{dataset}"
            ):
                img1_path = os.path.join(dataset_folder1, filename)
                img2_path = os.path.join(dataset_folder2, filename)

                try:
                    # Calculate CLIP similarity
                    clip_score = clip_similarity(img1_path, img2_path)

                    # Add result
                    current_max_id += 1
                    data["results"].append(
                        {
                            "id": current_max_id,
                            "Category": dataset,
                            "filename": filename,
                            "clip_score": float(clip_score),
                        }
                    )

                    processing_count += 1

                    # Save every 10 processed items
                    if processing_count % 10 == 0:
                        # Update category summary
                        update_category_summary(data, dataset)
                        save_json(data, json_path)

                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")

            # Update summary item for current dataset
            created_new = update_category_summary(data, dataset)
            if created_new:
                print(f"Created new summary item for {dataset}")

            # Save after processing each dataset
            save_json(data, json_path)


if __name__ == "__main__":
    # Source folder 1, containing models and datasets
    # Can be replaced when running
    source_folder1 = "pro_imgs"
    # Source folder 2, containing only datasets
    source_folder2 = "label_imgs"
    # Output directory
    output_dir = "clip"

    # Run comparison
    compare_model_datasets(source_folder1, source_folder2, output_dir)
