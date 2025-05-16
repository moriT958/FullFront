"""
Claude HTML Code Generator
=========================

This script processes data from parquet files and uses Claude API to generate HTML code from different inputs:
1. Text to code - Generate HTML from text descriptions
2. Image to code - Generate HTML from images 
3. Code Refinement - Add features to existing HTML based on images
4. Interaction_Authoring - Generate HTML from before/after image pairs

Usage:
------
1. Set your Anthropic API key in the script
2. Place your parquet data files in the "datasets" folder with appropriate names
3. Run the script: python claude_code.py

The script will process all data and save results to the "results/claude" folder.
Results are saved incrementally to avoid data loss.
"""

import os
import json
import base64
import re
import pandas as pd
from io import BytesIO
from PIL import Image
from anthropic import Anthropic
from typing import List, Dict, Any, Optional

# Utility functions
def load_parquet_data(file_path):
    """Load data from a Parquet file."""
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
    """Extract HTML code from text, looking for content starting with <!DOCTYPE html> and ending with </html>."""
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
        return base64_string  # For Claude, return the original base64 string
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None

def generate_claude_response(client, model_name, messages, temperature, max_tokens, retries=2):
    """Call Claude API to generate a response, with retry mechanism."""
    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            print(response)
            return response.content[0].text
        except Exception as e:
            pass

def process_text_to_code(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    """Process text-to-code conversion data."""
    results = []

    # Create set of processed IDs to skip
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
            
            # Build API request messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            # Generate response
            original_response = generate_claude_response(
                client, 
                model_name, 
                messages=messages,
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

            # Save results incrementally
            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
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

            # Save results on error
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_image_to_code(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    """Process image-to-code conversion data."""
    results = []

    # Create set of processed IDs to skip
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
            # Get base64 image
            base64_image = item['Image']
            if not base64_image:
                raise ValueError("Image data is empty")
                
            prompt = item['Prompt']
            
            # Build image content object
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image
                }
            }
            
            # Build API request messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content
                    ]
                }
            ]
            
            try:
                # Generate response
                original_response = generate_claude_response(
                    client, 
                    model_name, 
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract HTML code
                html_response = extract_html_code(original_response)
                
                # Log if HTML code was successfully extracted
                if html_response != original_response:
                    print(f"ID {item_id}: Successfully extracted HTML code")
                    
            except Exception as api_error:
                error_msg = str(api_error)
                
                # If image size exceeds limit, convert to JPEG and retry
                if "image exceeds 5 MB maximum" in error_msg:
                    print(f"Image exceeds size limit, converting to JPEG: {item_id}")
                    try:
                        # Convert to JPEG
                        img_data = base64.b64decode(base64_image)
                        img = Image.open(BytesIO(img_data))
                        
                        # Convert to RGB (if RGBA)
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                            
                        # Save as JPEG
                        buffer = BytesIO()
                        img.save(buffer, format='JPEG', quality=85)
                        buffer.seek(0)
                            
                        # Convert to base64
                        jpeg_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            
                        # Update image content
                        image_content = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": jpeg_base64
                            }
                        }
                        
                        # Update messages
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    image_content
                                ]
                            }
                        ]
                        
                        # Retry API call
                        original_response = generate_claude_response(
                            client, 
                            model_name, 
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # Extract HTML code
                        html_response = extract_html_code(original_response)
                        
                        print(f"Successfully converted to JPEG and completed request: {item_id}")
                            
                    except Exception as jpeg_error:
                        print(f"JPEG conversion failed: {jpeg_error}")
                        raise Exception(f"Image size exceeds limit and JPEG conversion failed - {jpeg_error}")
                else:
                    # Rethrow other API errors
                    raise api_error

            result = {
                "Id": item_id,
                "Response": html_response,
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result)

            # Save results incrementally
            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
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

            # Save results on error
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_refinement_to_code(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    results = []

    # Create set of processed IDs to skip
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
            # Get base64 image
            base64_image = item['Image']
            if not base64_image:
                raise ValueError("Image data is empty")
                
            prompt = item['Prompt']
            input_html = item['Input_html']
            
            # Build prompt text
            prompt_text = f"{prompt}\nCode:\n{input_html}"
            
            # Build image content object
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image
                }
            }
            
            # Build API request messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        image_content
                    ]
                }
            ]
            
            # Generate response
            original_response = generate_claude_response(
                client, 
                model_name, 
                messages=messages,
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

            # Save results incrementally
            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saved results to {output_path}.")

        except Exception as api_error:
                error_msg = str(api_error)
                
                # If image size exceeds limit, convert to JPEG and retry
                if "image exceeds 5 MB maximum" in error_msg:
                    print(f"Image exceeds size limit, converting to JPEG: {item_id}")
                    try:
                        # Convert to JPEG
                        img_data = base64.b64decode(base64_image)
                        img = Image.open(BytesIO(img_data))
                        
                        # Convert to RGB (if RGBA)
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                            
                        # Save as JPEG
                        buffer = BytesIO()
                        img.save(buffer, format='JPEG', quality=85)
                        buffer.seek(0)
                            
                        # Convert to base64
                        jpeg_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            
                        # Update image content
                        image_content = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": jpeg_base64
                            }
                        }
                        
                        # Update messages
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    image_content
                                ]
                            }
                        ]
                        
                        # Retry API call
                        original_response = generate_claude_response(
                            client, 
                            model_name, 
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # Extract HTML code
                        html_response = extract_html_code(original_response)
                        
                        print(f"Successfully converted to JPEG and completed request: {item_id}")

                        # Add result
                        result = {
                            "Id": item_id,
                            "Response": html_response,
                            "Label_html": item.get('Label_html'),
                            "Category": item.get('Category'),
                            "Png_id": item.get('Png_id')
                        }
                        results.append(result)
                        
                    except Exception as jpeg_error:
                        print(f"JPEG conversion failed: {jpeg_error}")
                        html_response = f"Processing error: Image size exceeds limit and JPEG conversion failed - {jpeg_error}"
                        
                        # Add error result
                        result = {
                            "Id": item_id,
                            "Response": html_response,
                            "Label_html": item.get('Label_html'),
                            "Category": item.get('Category'),
                            "Png_id": item.get('Png_id')
                        }
                        results.append(result)
                
                else:
                    print(f"Error processing Id {item_id}: {api_error}")
                    result = {
                        "Id": item_id,
                        "Response": f"Processing error: {api_error}",
                        "Label_html": item.get('Label_html'),
                        "Category": item.get('Category'),
                        "Png_id": item.get('Png_id')
                    }
                    results.append(result)

                    # Save results on error
                    if output_path:
                        save_results_to_json(results, output_path)
                        print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def process_interaction_to_code(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    """Process interaction-to-code conversion data."""
    results = []

    # Create set of processed IDs to skip
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
            # Get two base64 images
            before_image = item['Before_image']
            after_image = item['After_image']
            
            # Check if images are valid
            if not before_image or not after_image:
                raise ValueError("Image data is empty")

            prompt = item['Prompt']
            
            # Build image content objects
            before_image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": before_image
                }
            }
            
            after_image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": after_image
                }
            }
            
            # Build API request messages - using two images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        before_image_content,
                        after_image_content
                    ]
                }
            ]
            
            try:
                # Generate response
                original_response = generate_claude_response(
                    client, 
                    model_name, 
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract HTML code
                html_response = extract_html_code(original_response)
                
                # Log if HTML code was successfully extracted
                if html_response != original_response:
                    print(f"ID {item_id}: Successfully extracted HTML code")
                    
            except Exception as api_error:
                error_msg = str(api_error)
                
                # If image size exceeds limit, convert both images to JPEG and retry
                if "image exceeds 5 MB maximum" in error_msg:
                    print(f"Image exceeds size limit, converting to JPEG: {item_id}")
                    try:
                        # Function: Convert image to JPEG
                        def convert_to_jpeg(base64_image):
                            img_data = base64.b64decode(base64_image)
                            img = Image.open(BytesIO(img_data))
                            
                            # Convert to RGB (if RGBA)
                            if img.mode == 'RGBA':
                                img = img.convert('RGB')
                                
                            # Save as JPEG
                            buffer = BytesIO()
                            img.save(buffer, format='JPEG', quality=85)
                            buffer.seek(0)
                                
                            # Return base64 encoding
                            return base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        # Convert both images
                        before_jpeg = convert_to_jpeg(before_image)
                        after_jpeg = convert_to_jpeg(after_image)
                        
                        # Update image content
                        before_image_content = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": before_jpeg
                            }
                        }
                        
                        after_image_content = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": after_jpeg
                            }
                        }
                        
                        # Update messages
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    before_image_content,
                                    after_image_content
                                ]
                            }
                        ]
                        
                        # Retry API call
                        original_response = generate_claude_response(
                            client, 
                            model_name, 
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # Extract HTML code
                        html_response = extract_html_code(original_response)
                        
                        print(f"Successfully converted to JPEG and completed request: {item_id}")
                            
                    except Exception as jpeg_error:
                        print(f"JPEG conversion failed: {jpeg_error}")
                        raise Exception(f"Image size exceeds limit and JPEG conversion failed - {jpeg_error}")
                else:
                    # Rethrow other API errors
                    raise api_error

            result = {
                "Id": item_id,
                "Interaction_type": item.get('Interaction_type'),
                "Response": html_response,
                "Label_html": item.get('Label_html'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result)

            # Save results incrementally
            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
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

            # Save results on error
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Id {item_id}, saved current results to {output_path}.")

    # Ensure final results are saved
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Completed processing file, final results saved to: {output_path}")

    return results

def main(data_folder, api_key, model_name, output_base_dir, categories=None, max_tokens=8000, temperature=0):
    """Main function to select and run appropriate processing methods for each category."""
    # Ensure base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # All supported task types and their corresponding processing functions
    task_handlers = {
        "Text_to_code": process_text_to_code,
        "Image_to_code": process_image_to_code,
        "Code_Refinement": process_refinement_to_code,
        "Interaction_Authoring": process_interaction_to_code
    }

    # Validate specified categories
    unsupported_categories = [cat for cat in categories if cat not in task_handlers]
    if unsupported_categories:
        print(f"Warning: The following categories are not supported: {unsupported_categories}")
        # Filter out unsupported categories
        categories = [cat for cat in categories if cat in task_handlers]
        if not categories:
            print("No valid categories to process.")
            return

    print(f"Will process the following categories: {categories}")

    # --- Initialize Claude API client ---
    print("Initializing Claude API client...")
    try:
        # Instantiate Claude API client
        client = Anthropic(api_key=api_key)
        print("Claude API client initialization complete.")
    except Exception as e:
        print(f"Error initializing Claude API client: {e}")
        return

    # --- Process each selected category ---
    for category in categories:
        print("=" * 50)
        print(f"Starting to process category: {category}")
        
        # Find parquet file for corresponding category
        target_file = os.path.join(data_folder, f"{category}.parquet")
        
        if not os.path.exists(target_file):
            print(f"Cannot find file for category {category}: {target_file}")
            continue
            
        print(f"Found category file: {target_file}")
        
        # Generate output filename
        output_filename = f"claude_{category}.json"
        current_output_path = os.path.join(output_base_dir, output_filename)
        print(f"Results will be saved to: {current_output_path}")
        
        # Load existing results for current file
        existing_results = load_existing_results(current_output_path)
        
        # Load data
        data_list = load_parquet_data(target_file)
        
        if not data_list:
            print(f"No data in file {target_file} or loading failed.")
            continue
            
        # Get corresponding processing function
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
        
        print(f"Category {category} processing complete.")

    print("=" * 50)
    print("All specified categories processing complete.")

if __name__ == "__main__":
    data_folder = "./datasets"  # Data folder path
    api_key = "your_api_key"
    model_name = "claude-3-7-sonnet-20250219"  # Claude model
    output_dir = "./results/claude"

    # Specify which dataset categories to process
    categories = ["Text_to_code", "Image_to_code", "Code_Refinement", "Interaction_Authoring"]

    max_tokens = 20000 
    temperature = 0

    # Call main function
    main(data_folder, api_key, model_name, output_dir, categories, max_tokens, temperature)