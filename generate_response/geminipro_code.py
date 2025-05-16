"""
Gemini Pro Code Generation Tool

Usage:
1. Install required dependencies: 
   pip install pandas google-generativeai pillow 

2. Set your Gemini API key in the script or as an environment variable.

3. Prepare your dataset files in the './datasets' directory in parquet format.

4. Run the script:
   python geminipro_code.py

This script processes different types of tasks (text-to-code, image-to-code, etc.)
using the Gemini API and saves the results to JSON files.
"""

import os
import json
import base64
import glob
import re
import pandas as pd
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
import time
from typing import List, Dict, Any, Optional

# Common functions
def load_parquet_data(file_path):
    """Load data from a single Parquet file."""
    try:
        df = pd.read_parquet(file_path)
        return df.to_dict('records')
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return []

def load_existing_results(output_path):
    """Load existing results file to skip already processed data."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load existing results file {output_path}: {e}")
    return []

def save_results_to_json(results, output_path="output.json"):
    """Save results to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save results to {output_path}: {e}")

def extract_html_code(text):
    """Extract HTML code from text, starting with <!DOCTYPE html> and ending with </html>.
    If no matching HTML code is found, return the original text."""
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
    """Decode base64 encoded image to bytes."""
    try:
        image_bytes = base64.b64decode(base64_string)
        return image_bytes
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None

def generate_gemini_response(client, model_name, contents, temperature, max_tokens, retries=2):
    """Call Gemini API to generate response, with retry mechanism."""
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=thinking_config,
                    max_output_tokens=max_tokens
                )
            )
            print(response)
            return response.text.strip()
        except Exception as e:
            if attempt < retries:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"API call failed, retrying in {wait_time} seconds: {e}")
                time.sleep(wait_time)
            else:
                raise Exception(f"API call failed after {retries+1} attempts: {e}")

def process_text_to_code(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    """Process text-to-code conversion data."""
    results = []

    # Create a set of processed item IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy()
        print(f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        # Skip already processed items
        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            prompt = item['Prompt']
            input_text = item['Input_text']
            
            # Build prompt text
            prompt_text = f"{prompt}\nDescription:{input_text}"
            
            # Generate response
            original_response = generate_gemini_response(
                client, 
                model_name, 
                contents=prompt_text,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract HTML code
            html_response = extract_html_code(original_response)
            
            # Log if HTML code was successfully extracted
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

            # Save results every 5 new items or at the last item
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

            # Save results on error
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_image_to_code(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    """Process image-to-code conversion data."""
    results = []

    # Create a set of processed item IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy()
        print(f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        # Skip already processed items
        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            # Decode base64 image
            image_bytes = decode_base64_image(item['Image'])
            if image_bytes is None:
                raise ValueError("Image decoding failed")
                
            prompt = item['Prompt']
            
            # Build API request content
            contents = [
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]
            
            # Generate response
            original_response = generate_gemini_response(
                client, 
                model_name, 
                contents=contents,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract HTML code
            html_response = extract_html_code(original_response)
            
            # Log if HTML code was successfully extracted
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

            # Save results every 5 new items or at the last item
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

            # Save results on error
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_refinement_to_code(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    results = []

    # Create a set of processed item IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy()
        print(f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        # Skip already processed items
        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            # Decode base64 image
            image_bytes = decode_base64_image(item['Image'])
            if image_bytes is None:
                raise ValueError("Image decoding failed")
                
            prompt = item['Prompt']
            input_html = item['Input_html']
            
            # Build prompt text
            prompt_text = f"{prompt}\nCode:\n{input_html}"
            
            # Build API request content
            contents = [
                prompt_text,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]
            
            # Generate response
            original_response = generate_gemini_response(
                client, 
                model_name, 
                contents=contents,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract HTML code
            html_response = extract_html_code(original_response)
            
            # Log if HTML code was successfully extracted
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

            # Save results every 5 new items or at the last item
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

            # Save results on error
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_interaction_to_code(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    """Process interaction-to-code conversion data."""
    results = []

    # Create a set of processed item IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy()
        print(f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        # Skip already processed items
        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            # Decode two base64 images
            before_image_bytes = decode_base64_image(item['Before_image'])
            after_image_bytes = decode_base64_image(item['After_image'])
            
            # Check if images were decoded correctly
            if before_image_bytes is None or after_image_bytes is None:
                raise ValueError("Failed to decode images")

            prompt = item['Prompt']
            
            # Build API request content - using two images
            contents = [
                prompt,
                types.Part.from_bytes(data=before_image_bytes, mime_type="image/jpeg"),
                types.Part.from_bytes(data=after_image_bytes, mime_type="image/jpeg")
            ]
            
            # Generate response
            original_response = generate_gemini_response(
                client, 
                model_name, 
                contents=contents,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract HTML code
            html_response = extract_html_code(original_response)
            
            # Log if HTML code was successfully extracted
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

            # Save results every 5 new items or at the last item
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

            # Save results on error
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def main(data_folder, api_key, model_name, output_base_dir, categories=None, max_tokens=8000, temperature=0):
    """Main function to process data based on specified categories."""
    # Ensure base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # All supported task types and their corresponding processing functions
    task_handlers = {
        "Text_to_code": process_text_to_code,
        "Image_to_code": process_image_to_code,
        "Code_Refinement": process_refinement_to_code,
        "Interaction_Authoring": process_interaction_to_code
    }

    # If no categories specified, process all supported categories
    if categories is None:
        categories = list(task_handlers.keys())

    # Validate if specified categories are supported
    unsupported_categories = [cat for cat in categories if cat not in task_handlers]
    if unsupported_categories:
        print(f"Warning: The following categories are not supported: {unsupported_categories}")
        # Filter out unsupported categories
        categories = [cat for cat in categories if cat in task_handlers]
        if not categories:
            print("No valid categories to process.")
            return

    print(f"Will process the following categories: {categories}")

    # --- Initialize Gemini API client ---
    print("Initializing Gemini API client...")
    try:
        # Instantiate Gemini API client
        client = genai.Client(api_key=api_key)
        print("Gemini API client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini API client: {e}")
        return

    # --- Process each selected category ---
    for category in categories:
        print("=" * 50)
        print(f"Starting to process category: {category}")
        
        # Find corresponding parquet file for the category
        target_file = os.path.join(data_folder, f"{category}.parquet")
        
        if not os.path.exists(target_file):
            print(f"Could not find file for category {category}: {target_file}")
            continue
            
        print(f"Found category file: {target_file}")
        
        # Generate output filename
        output_filename = f"gemini_{category}.json"
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
        
        # Process data
        process_func(
            data_list, 
            client, 
            model_name, 
            temperature, 
            max_tokens, 
            existing_results=existing_results,
            output_path=current_output_path
        )
        
        print(f"Category {category} processing completed.")

    print("=" * 50)
    print("All specified categories processing completed.")

if __name__ == "__main__":
    data_folder = "./datasets"  # Data folder path
    api_key = "your_api_key"  # Gemini API key
    model_name = "gemini-2.5-pro-preview-05-06"  # Gemini model name
    output_dir = "./results/gemini"  # Directory to save results

    # Specify categories to process
    categories = ["Text_to_code", "Image_to_code", "Code_Refinement", "Interaction_Authoring"]

    max_tokens = 20000  # Maximum tokens to generate
    temperature = 0  # Temperature parameter
    thinking_config = types.ThinkingConfig(thinking_budget=0)

    # Call main function
    main(data_folder, api_key, model_name, output_dir, categories, max_tokens, temperature)