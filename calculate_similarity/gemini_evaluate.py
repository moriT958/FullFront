"""
# Gemini Image Comparison Evaluator

This script evaluates the similarity between pairs of webpage images using Google's Gemini model.
It compares model-generated images with ground truth images across multiple dimensions and saves 
the evaluation scores as JSON files.

## Usage:
1. Place model-generated images in the 'o4mini_imgs' directory
2. Place ground truth images in the 'label_imgs' directory  
3. Add your Gemini API key(s) to the api_keys list
4. Run the script: python gemini_evaluate.py

## Directory Structure:
- o4mini_imgs/[model]/[dataset]/*.png - Model generated images
- label_imgs/[dataset]/*.png - Ground truth images
- results/ - Output directory for evaluation results
"""

from google import genai
from google.genai import types
import os
import glob
import json
import re
import time
from pathlib import Path
from datetime import datetime

# API key configuration
api_keys = [
    "your keys"
]

current_api_key_index = 0 
api_key = api_keys[current_api_key_index]
client = genai.Client(api_key=api_key)

def switch_to_next_api_key():
    global current_api_key_index, client, api_key
    current_api_key_index = (current_api_key_index + 1) % len(api_keys)
    api_key = api_keys[current_api_key_index]
    client = genai.Client(api_key=api_key)
    print(f"Switching to new API key: {current_api_key_index + 1}/{len(api_keys)}")
    write_log(f"Switched to new API key index {current_api_key_index + 1}/{len(api_keys)}")
    return api_key

# Source folders and output path
source_folder1 = "o4mini_imgs"  # Model generated images
source_folder2 = "label_imgs"   # Ground truth images
json_save_path = "./results"    # JSON save path

os.makedirs(json_save_path, exist_ok=True)

models = ["llava", "o1", "openai", "gemini", "internvl2", "internvl3", "qwen", "claude", "o4mini", "pro"] # models
datasets = ["Code_Refinement", "Image_to_code", "Interaction_Authoring", "Text_to_code"]

prompt = '''
Your task is to assess two webpage images and output a score between 0 and 10 for each of the following 10 questions, reflecting the degree of similarity between the webpages. A score of 10 indicates perfect similarity (identical in every aspect), while a score of 0 indicates no similarity. For partial similarities, assign a score between 1 and 9, where higher scores reflect greater similarity. Only output a comma-separated list of 10 numbers enclosed in square brackets, e.g., [10,8,6,4,2,0,0,0,0,0]. Do not assign a score of 10 unless the two images are identical in the respective category.

1. **Element Reproduction (Score: 0-10):** Are key elements (text, images, buttons) fully present and styled identically to the original design? (e.g., 10 for identical elements, 5 for missing or slightly altered elements, 0 for completely different elements.)
2. **Proportion and Size Consistency (Score: 0-10):** Do the sizes and proportions of elements (text, images, buttons) match the original design, maintaining visual harmony? (e.g., 10 for exact proportions, 6 for minor size differences, 0 for significant discrepancies.)
3. **Layout and Typography Fidelity (Score: 0-10):** Does the overall layout (headers, footers, navigation bars, sidebars) faithfully replicate the original design's typography and structure? (e.g., 10 for identical layouts, 5 for similar but not exact placements, 0 for entirely different layouts.)
4. **Alignment and Spacing Accuracy (Score: 0-10):** Are elements aligned and spaced (margins, padding) as in the original design? (e.g., 10 for perfect alignment and spacing, 6 for minor misalignments, 0 for major misalignments.)
5. **Visual Hierarchy Clarity (Score: 0-10):** Does the webpage maintain the same visual hierarchy as the original, allowing users to quickly identify key information? (e.g., 10 for identical hierarchy, 5 for slightly altered emphasis, 0 for unclear or different hierarchy.)
6. **Color Consistency (Score: 0-10):** Do the overall color scheme, hues, and tones match the original design? (e.g., 10 for identical colors, 6 for similar palette with minor variations, 0 for completely different colors.)
7. **Style Consistency (Score: 0-10):** Does the webpage's overall aesthetic style (e.g., modern, minimalistic) align with the original design? (e.g., 10 for identical style, 4 for similar but distinguishable style, 0 for entirely different style.)
8. **Text Style Consistency (Score: 0-10):** Are text attributes (font type, size, line spacing, paragraph spacing, alignment) consistent with the original design? (e.g., 10 for identical text styles, 5 for similar fonts with spacing issues, 0 for completely different text styles.)
9. **Text Content Accuracy (Score: 0-10):** Does the webpage accurately reproduce the main textual content of the original design? (e.g., 10 for identical text, 5 for partial matches, 0 for entirely different text.)
10. **Overall Content Representation (Score: 0-10):** Does the webpage convey the same content information and intent as the original design? (e.g., 10 for identical content representation, 6 for similar but incomplete content, 0 for entirely different content.)

**Output Format:** [score1,score2,score3,score4,score5,score6,score7,score8,score9,score10]
'''

log_file = os.path.join(json_save_path, "log.txt")

def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def update_category_summary(model, category, model_data):
    category_items = [item for item in model_data[model]["items"] 
                     if item.get("Category") == category and not item.get("id", "").endswith("_summary")]
    
    if not category_items:
        return
    
    all_scores = [item["scores"] for item in category_items]
    
    avg_scores = []
    for i in range(10):  
        dimension_scores = [scores[i] for scores in all_scores]
        avg_scores.append(round(sum(dimension_scores) / len(dimension_scores), 2))
    
    summary_id = f"{category}_summary"
    summary_exists = False
    
    for i, item in enumerate(model_data[model]["items"]):
        if item.get("id") == summary_id:
            model_data[model]["items"][i]["scores"] = avg_scores
            summary_exists = True
            break
    
    if not summary_exists:
        summary_item = {
            "id": summary_id,
            "Category": category,
            "scores": avg_scores
        }
        model_data[model]["items"].append(summary_item)

model_data = {}
for model in models:
    json_file = os.path.join(json_save_path, f"{model}.json")
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                model_data[model] = json.load(f)
            except json.JSONDecodeError:
                model_data[model] = {"items": []}
    else:
        model_data[model] = {"items": []}

total_pairs = 0
processed_pairs = 0
model_processed_counts = {model: 0 for model in models}

try:
    for model in models:
        for dataset in datasets:
            dataset_path1 = os.path.join(source_folder1, model, dataset)
            dataset_path2 = os.path.join(source_folder2, dataset)
            
            if not os.path.exists(dataset_path1) or not os.path.exists(dataset_path2):
                message = f"Path does not exist: {dataset_path1} or {dataset_path2}"
                print(message)
                write_log(message)
                continue
            
            images1 = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images1.extend(glob.glob(os.path.join(dataset_path1, ext)))
            
            for img1_path in images1:
                img_name = os.path.basename(img1_path)
                img2_path = os.path.join(dataset_path2, img_name)
                
                if os.path.exists(img2_path):
                    img_id = f"{dataset}_{Path(img_name).stem}"
                    already_processed = False
                    
                    for item in model_data[model]["items"]:
                        if item.get("id") == img_id:
                            already_processed = True
                            break
                    
                    if already_processed:
                        print(f"Skipping already processed image: {model}/{dataset}/{img_name}")
                        continue
                    
                    total_pairs += 1
                    
                    with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2:
                        img1_bytes = f1.read()
                        img2_bytes = f2.read()
            
                    success = False
                    retry_count = 0
                    max_retries = 2
                    error_messages = []
                    
                    while not success and retry_count <= max_retries:
                        try:
                            if retry_count > 0:
                                print(f"Retry #{retry_count} processing image: {model}/{dataset}/{img_name}")
                            
                            response = client.models.generate_content(
                                model="gemini-2.5-flash-preview-04-17",
                                contents=[
                                    prompt,
                                    types.Part.from_bytes(data=img1_bytes, mime_type='image/png'),
                                    types.Part.from_bytes(data=img2_bytes, mime_type='image/png')
                                ]
                            )
                            
                            response_text = response.text
                            scores_match = re.search(r'\[(.*?)\]', response_text)
                            if scores_match:
                                scores_text = scores_match.group(1)
                                scores = [float(s.strip()) for s in scores_text.split(',')]
                                if len(scores) == 10:
                                    item_data = {
                                        "id": img_id,
                                        "Category": dataset,
                                        "scores": scores
                                    }
                                    model_data[model]["items"].append(item_data)
                                    
                                    update_category_summary(model, dataset, model_data)
                                    
                                    processed_pairs += 1
                                    model_processed_counts[model] += 1
                                    print(f"Processed image pair {processed_pairs}/{total_pairs}: {model}/{dataset}/{img_name}")
                                    
                                    if model_processed_counts[model] % 1 == 0:
                                        json_file = os.path.join(json_save_path, f"{model}.json")
                                        with open(json_file, 'w', encoding='utf-8') as f:
                                            json.dump(model_data[model], f, indent=2, ensure_ascii=False)
                                        print(f"Saved {model}.json (processed {model_processed_counts[model]} images)")
                                    success = True
                                else:
                                    error_msg = f"Did not get 10 scores: {scores}"
                                    error_messages.append(error_msg)
                                    print(f"Warning: {model}/{dataset}/{img_name} {error_msg}")
                            else:
                                error_msg = f"Could not extract score list from response: {response_text}"
                                error_messages.append(error_msg)
                                print(f"Warning: {model}/{dataset}/{img_name} {error_msg}")
                        
                        except Exception as e:
                            error_msg = f"Error processing image: {str(e)}"
                            error_messages.append(error_msg)
                            print(f"Warning: {model}/{dataset}/{img_name} {error_msg}")
                            
                            error_str = str(e)
                            if "RESOURCE_EXHAUSTED" in error_str and "exceeded your current quota" in error_str:
                                switch_to_next_api_key()
                                continue
                        
                        retry_count += 1
                        if not success and retry_count <= max_retries:
                            time.sleep(2)
                    
                    if not success:
                        error_log = f"Failed to process image (after {max_retries+1} attempts): {model}/{dataset}/{img_name}\n"
                        for i, msg in enumerate(error_messages):
                            error_log += f"  Attempt {i+1} error: {msg}\n"
                        write_log(error_log)
                
except Exception as e:
    message = f"Error occurred, stopping script: {str(e)}"
    print(message)
    write_log(message)
    for model in models:
        if model in model_data:
            for dataset in datasets:
                update_category_summary(model, dataset, model_data)
            
            json_file = os.path.join(json_save_path, f"{model}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(model_data[model], f, indent=2, ensure_ascii=False)

for model in models:
    if model in model_data:
        for dataset in datasets:
            update_category_summary(model, dataset, model_data)
        
        json_file = os.path.join(json_save_path, f"{model}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(model_data[model], f, indent=2, ensure_ascii=False)

print(f"Complete! Processed {processed_pairs}/{total_pairs} image pairs.")