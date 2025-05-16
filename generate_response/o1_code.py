"""
O1 Model HTML Code Generation Tool

Usage:
1. Set your OpenAI API key in the main() function
2. Prepare your data in the './mini_datasets' folder (parquet files)
3. Specify categories to process in the main() function
4. Run the script: python o1_code.py

This script processes text and image inputs to generate HTML code using OpenAI's models.
"""

import os
import json
import base64
import re
import pandas as pd
from openai import OpenAI
import time
from typing import List, Dict, Any, Optional

# Utility functions
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
    """Save results to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save results to {output_path}: {e}")

def extract_html_code(text):
    """Extract HTML code from text, requiring it to start with <!DOCTYPE html> and end with </html>.
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
    """Keep base64 encoded image as base64 format for API calls."""
    try:
        # Validate base64 string
        base64.b64decode(base64_string)
        return base64_string  # For OpenAI, we return the original base64 string
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None

def generate_openai_response(client, model_name, messages, retries=2):
    """Call OpenAI API to generate a response, with retry mechanism."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"API call failed, retrying in {wait_time} seconds: {e}")
                time.sleep(wait_time)
            else:
                raise Exception(f"API call failed after {retries+1} attempts: {e}")

def process_text_to_code(data_list, client, model_name, existing_results=None, output_path=None):
    """Process text-to-code conversion data."""
    results = []

    # Create a set of processed question IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Id') for item in existing_results if item.get('Id')}
        results = existing_results.copy()  # Use existing results for this file
        print(f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get('Id')

        # Skip already processed questions
        if item_id in processed_ids:
            continue

        newly_processed_count += 1  # Count newly processed questions

        try:
            prompt = item['Prompt']
            input_text = item['Input_text']
            
            # Build prompt text
            prompt_text = f"{prompt}\nDescription:{input_text}"
            
            # Build API request messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            # Generate reply
            original_response = generate_openai_response(
                client, 
                model_name, 
                messages=messages,
            )
            
            # Extract HTML code
            html_response = extract_html_code(original_response)
            
            # If HTML code was successfully extracted, log it
            if html_response != original_response:
                print(f"ID {item_id}: Successfully extracted HTML code")

            result = {
                "Id": item_id,
                "Response": html_response,
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result)  # Add new result to list

            # Save results after processing each new item or at the end
            if newly_processed_count > 0 and (newly_processed_count % 1 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saved results to {output_path}.")

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Response": f"Processing error: {e}",  # Record error information
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }
            results.append(result)

            # Save results when an error occurs
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved at the end of the function if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_image_to_code(data_list, client, model_name, existing_results=None, output_path=None):
    """Process image-to-code conversion data."""
    results = []

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
            if not base64_image:
                raise ValueError("Image data is empty")
                
            prompt = item['Prompt']
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            original_response = generate_openai_response(
                client, 
                model_name, 
                messages=messages,
            )
            
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

            if newly_processed_count > 0 and (newly_processed_count % 1 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saved results to {output_path}.")

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
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_refinement_to_code(data_list, client, model_name, existing_results=None, output_path=None):
    results = []

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
            if not base64_image:
                raise ValueError("Image data is empty")
                
            prompt = item['Prompt']
            input_html = item['Input_html']
            
            prompt_text = f"{prompt}\nCode:\n{input_html}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            original_response = generate_openai_response(
                client, 
                model_name, 
                messages=messages,
            )
            
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

            if newly_processed_count > 0 and (newly_processed_count % 1 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saved results to {output_path}.")

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
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_interaction_to_code(data_list, client, model_name, existing_results=None, output_path=None):
    """Process interaction-to-code conversion data."""
    results = []

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
            before_image = item['Before_image']
            after_image = item['After_image']
            
            if not before_image or not after_image:
                raise ValueError("Image data is empty")

            prompt = item['Prompt']
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{before_image}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{after_image}"
                            }
                        }
                    ]
                }
            ]
            
            original_response = generate_openai_response(
                client, 
                model_name, 
                messages=messages,
            )
            
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

            if newly_processed_count > 0 and (newly_processed_count % 1 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saved results to {output_path}.")

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
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def main(data_folder, api_key, model_name, output_base_dir, categories=None):
    """Main function that selects the appropriate processing method based on specified categories."""
    # Ensure the base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # All supported task types and corresponding processing functions
    task_handlers = {
        "Text_to_code_mini": process_text_to_code,
        "Image_to_code_mini": process_image_to_code,
        "Code_Refinement_mini": process_refinement_to_code,
        "Interaction_Authoring_mini": process_interaction_to_code
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

    # Initialize OpenAI API client
    print("Initializing OpenAI API client...")
    try:
        client = OpenAI(api_key=api_key)
        print("OpenAI API client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI API client: {e}")
        return

    # Process each selected category
    for category in categories:
        print("=" * 50)
        print(f"Starting to process category: {category}")
        
        # Find the corresponding parquet file for this category
        target_file = os.path.join(data_folder, f"{category}.parquet")
        
        if not os.path.exists(target_file):
            print(f"Could not find file for category {category}: {target_file}")
            continue
            
        print(f"Found category file: {target_file}")
        
        # Generate output filename
        output_filename = f"o1_{category}.json"
        current_output_path = os.path.join(output_base_dir, output_filename)
        print(f"Results will be saved to: {current_output_path}")
        
        # Load existing results for this file
        existing_results = load_existing_results(current_output_path)
        
        # Load data
        data_list = load_parquet_data(target_file)
        
        if not data_list:
            print(f"No data in file {target_file} or failed to load.")
            continue
            
        # Get the corresponding processing function
        process_func = task_handlers[category]
        
        # Process the data
        process_func(
            data_list, 
            client, 
            model_name, 
            existing_results=existing_results,
            output_path=current_output_path
        )
        
        print(f"Category {category} processing complete.")

    print("=" * 50)
    print("All specified category processing workflows completed.")

if __name__ == "__main__":
    data_folder = "./mini_datasets"  # Data folder path
    api_key = "your_api_key"  # Replace with your actual API key
    model_name = "o1"
    output_dir = "./results/o1"  # Directory for saving results

    # Specify which categories to process
    categories = ["Text_to_code_mini", "Image_to_code_mini", "Code_Refinement_mini", "Interaction_Authoring_mini"]

    # Call the main function
    main(data_folder, api_key, model_name, output_dir, categories)