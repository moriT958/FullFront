"""
InternVL2.5 Code Generation Tool

Usage:
    python internvl2_5_code.py [--data_folder DATA_FOLDER] [--model_path MODEL_PATH] 
                             [--output_dir OUTPUT_DIR] [--categories CATEGORIES] 
                             [--max_tokens MAX_TOKENS] [--temperature TEMPERATURE] 
                             [--tensor_parallel_size TP_SIZE]
"""

import os
import json
import base64
import re
import pandas as pd
from io import BytesIO
from PIL import Image
from vllm import LLM, SamplingParams

def load_parquet_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        return df.to_dict('records')
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return []

def load_existing_results(output_path):
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load existing results file {output_path}: {e}")
    return []

def save_results_to_json(results, output_path="output.json"):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save results to {output_path}: {e}")

def extract_html_code(text):
    html_pattern = r'(?:<\!DOCTYPE\s+html>.*?<\/html>)'
    matches = re.findall(html_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0]
    if '<!DOCTYPE html>' in text.lower() and '</html>' in text.lower():
        start = text.lower().find('<!doctype html>')
        end = text.lower().find('</html>') + len('</html>')
        return text[start:end]
    return text

def decode_base64_image(base64_string):
    try:
        image_bytes = base64.b64decode(base64_string)
        image_io = BytesIO(image_bytes)
        image = Image.open(image_io)
        return image
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None

def process_text_to_code(data_list, llm, temperature, max_tokens, existing_results=None, output_path=None):
    results = []
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy() 
        print(f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            prompt = item['Prompt']
            input_text = item['Input_text']

            internvl_prompt = f"USER: {prompt}\nDescription:{input_text}\nASSISTANT:"

            inputs = {
                "prompt": internvl_prompt,
            }

            outputs = llm.generate([inputs], sampling_params=sampling_params)
            original_response = outputs[0].outputs[0].text.strip()
            
            html_response = extract_html_code(original_response)
            
            if html_response != original_response:
                print(f"ID {item_id}: Successfully extracted HTML code")

            result = {
                "Id": item_id,
                "Response": html_response,
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result) 

            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} items (new: {newly_processed_count}), saved results to {output_path}.")

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Response": f"Processing error: {e}",
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }
            results.append(result)

            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"File processing complete, final results saved to: {output_path}")

    return results

def process_image_to_code(data_list, llm, temperature, max_tokens, existing_results=None, output_path=None):
    results = []
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy()
        print(f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        if item_id in processed_ids:
            continue

        newly_processed_count += 1 

        try:
            base64_image = item['Image']
            image_bytes = base64.b64decode(base64_image)
            image_io = BytesIO(image_bytes)
            image = Image.open(image_io)

            prompt = item['Prompt']

            internvl_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

            inputs = {
                "prompt": internvl_prompt,
                "multi_modal_data": {
                    "image": image
                },
            }

            outputs = llm.generate([inputs], sampling_params=sampling_params)
            original_response = outputs[0].outputs[0].text.strip()
            
            html_response = extract_html_code(original_response)
            
            if html_response != original_response:
                print(f"ID {item_id}: Successfully extracted HTML code")

            result = {
                "Id": item_id,
                "Response": html_response,
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result) 

            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} items (new: {newly_processed_count}), saved results to {output_path}.")

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Response": f"Processing error: {e}",
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }
            results.append(result)

            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"File processing complete, final results saved to: {output_path}")

    return results

def process_refinement_to_code(data_list, llm, temperature, max_tokens, existing_results=None, output_path=None):
    results = []
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy() 

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        if item_id in processed_ids:
            continue

        newly_processed_count += 1 

        try:
            base64_image = item['Image']
            image_bytes = base64.b64decode(base64_image)
            image_io = BytesIO(image_bytes)
            image = Image.open(image_io)

            prompt = item['Prompt']
            input_html = item['Input_html']

            internvl_prompt = f"USER: <image>\n{prompt}\nCode:\n{input_html}\nASSISTANT:"

            inputs = {
                "prompt": internvl_prompt,
                "multi_modal_data": {
                    "image": image
                },
            }

            outputs = llm.generate([inputs], sampling_params=sampling_params)
            original_response = outputs[0].outputs[0].text.strip()
            
            html_response = extract_html_code(original_response)
            
            result = {
                "Id": item_id,
                "Response": html_response,
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result) 
            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Response": f"Processing error: {e}",
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }
            results.append(result)

            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"File processing complete, final results saved to: {output_path}")

    return results

def process_interaction_to_code(data_list, llm, temperature, max_tokens, existing_results=None, output_path=None):
    results = []
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy() 
        print(f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            before_image = decode_base64_image(item['Before_image'])
            after_image = decode_base64_image(item['After_image'])
            
            if before_image is None or after_image is None:
                raise ValueError("Failed to decode images")

            prompt = item['Prompt']

            internvl_prompt = f"USER: <image>\n<image>\n{prompt}\nASSISTANT:"

            inputs = {
                "prompt": internvl_prompt,
                "multi_modal_data": {
                    "image": [before_image, after_image]
                },
            }

            outputs = llm.generate([inputs], sampling_params=sampling_params)
            original_response = outputs[0].outputs[0].text.strip()
            
            html_response = extract_html_code(original_response)
            
            if html_response != original_response:
                print(f"ID {item_id}: Successfully extracted HTML code")

            result = {
                "Id": item_id,
                "Interaction_type": item.get('Interaction_type'),
                "Response": html_response,
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result)

            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} items (new: {newly_processed_count}), saved results to {output_path}.")

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Interaction_type": item.get('Interaction_type'),
                "Response": f"Processing error: {e}",
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }
            results.append(result)

            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"File processing complete, final results saved to: {output_path}")

    return results

def main(data_folder="./data", model_path="./models/InternVL2_5", output_base_dir="./results", 
         categories=None, max_tokens=256, temperature=0, tensor_parallel_size=8):
    """Main function that selects the appropriate processing method based on the specified categories."""
    # Ensure the base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # All supported task types and their corresponding processing functions
    task_handlers = {
        "Text_to_code": process_text_to_code,
        "Image_to_code": process_image_to_code,
        "Code Refinement": process_refinement_to_code,
        "Interaction_Authoring": process_interaction_to_code
    }

    # If no categories are specified, default to processing all supported categories
    if categories is None:
        categories = list(task_handlers.keys())

    # Validate that all specified categories are supported
    unsupported_categories = [cat for cat in categories if cat not in task_handlers]
    if unsupported_categories:
        print(f"Warning: The following categories are not supported: {unsupported_categories}")
        # Filter out unsupported categories
        categories = [cat for cat in categories if cat in task_handlers]
        if not categories:
            print("No valid categories to process.")
            return

    print(f"Will process the following categories: {categories}")

    # --- Model Loading ---
    print("Loading model...")
    try:
        # Determine if multimodal tasks need to be processed
        multimodal_tasks = ["Image_to_code", "Code_Refinemnet", "Interaction_Authoring", "Text_to_code"]
        requires_multimodal = any(cat in multimodal_tasks for cat in categories)
        
        # Create LLM configuration
        llm_config = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": 32768,
            "enforce_eager": True
        }
        
        # For multimodal tasks, add multimodal processing configuration
        if requires_multimodal:
            # Determine if multi-image tasks need to be processed
            multi_image_tasks = ["Interaction_Authoring"]
            max_images = 2 if any(cat in multi_image_tasks for cat in categories) else 1
            llm_config["limit_mm_per_prompt"] = {"image": max_images, "video": 0}
        
        llm = LLM(**llm_config)
        print("Model loading complete.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Process each selected category ---
    for category in categories:
        print("=" * 50)
        print(f"Starting to process category: {category}")
        
        # Find the corresponding parquet file for the category
        target_file = os.path.join(data_folder, f"{category}.parquet")
        
        if not os.path.exists(target_file):
            print(f"Could not find file for category {category}: {target_file}")
            continue
            
        print(f"Found category file: {target_file}")
        
        # Generate output filename
        output_filename = f"internvl2_5_{category}.json"
        current_output_path = os.path.join(output_base_dir, output_filename)
        print(f"Results will be saved to: {current_output_path}")
        
        # Load existing results for the current file
        existing_results = load_existing_results(current_output_path)
        
        # Load data
        data_list = load_parquet_data(target_file)
        
        if not data_list:
            print(f"No data in file {target_file} or loading failed.")
            continue
            
        # Get the corresponding processing function
        process_func = task_handlers[category]
        
        # Process the data
        process_func(
            data_list, 
            llm, 
            temperature, 
            max_tokens, 
            existing_results=existing_results,
            output_path=current_output_path
        )
        
        print(f"Processing for category {category} completed.")

    print("=" * 50)
    print("Processing workflow for all specified categories is complete.")

if __name__ == "__main__":
    # Default paths - adjust these to relative paths for your environment
    data_folder = "./data"
    model_path = "./models/InternVL2_5"
    output_dir = "./results"

    # Specify which dataset categories to process
    categories = ["Text_to_code", "Image_to_code", "Code_Refinemnet", "Interaction_Authoring"]

    max_tokens = 30000  # Maximum number of tokens to generate
    temperature = 0  # Temperature parameter
    tensor_parallel_size = 4  # Number of GPUs for parallelism

    # Call the main function
    main(data_folder, model_path, output_dir, categories, max_tokens, temperature, tensor_parallel_size)
