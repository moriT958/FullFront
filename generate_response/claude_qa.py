"""
Claude QA: A tool for evaluating Claude's performance on image-based question answering tasks.

Usage:
1. Place your parquet datasets in the ./datasets folder
2. Configure your API key and other parameters in the main section
3. Run: python claude_qa.py
4. Results will be saved in ./results/claude directory

The script processes image-based QA datasets, sends queries to Claude API, and saves the responses.
It has checkpoint functionality to resume from interrupted runs.
"""

import os
import json
import base64
import glob
import pandas as pd
from PIL import Image
from io import BytesIO
from anthropic import Anthropic
import time
import math

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


def process_data(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    """Process data list, generate model responses, and save results in real-time."""
    results = []
    
    # Create a set of processed question IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Question_id') for item in existing_results if item.get('Question_id')}
        results = existing_results.copy()  # Use existing results for this file
        print(f"Loaded {len(processed_ids)} already processed questions (from {output_path}), will skip these questions.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        question_id = item.get('Question_id')

        # Skip already processed questions
        if question_id in processed_ids:
            print(f"Skipping already processed question ID: {question_id}")
            continue

        newly_processed_count += 1  # Count newly processed questions

        try:
            # Decode base64 image
            base64_image = item['Image']
            
            prompt = item['Prompt']
            question = item['Question']
            choices = item['Choices']
            
            # Build prompt text
            prompt_text = f"{prompt}\nQuestion: {question}\nChoices: {choices}"
            
            # Build image content object
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image
                }
            }
            
            # Construct API request
            try:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                image_content
                            ]
                        }
                    ]
                )
                
                # Get response text
                model_response = response.content[0].text

            except Exception as api_error:
                error_msg = str(api_error)
                
                # If image size exceeds limit, convert to JPEG and retry
                if "image exceeds 5 MB maximum" in error_msg:
                    print(f"Image exceeds size limit, converting to JPEG: {question_id}")
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
                        
                        # Retry API call
                        response = client.messages.create(
                            model=model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt_text},
                                        image_content
                                    ]
                                }
                            ]
                        )
                        
                        model_response = response.content[0].text
                        print(f"Successfully converted to JPEG and completed request: {question_id}")
                        
                    except Exception as jpeg_error:
                        print(f"JPEG conversion failed: {jpeg_error}")
                        model_response = f"Processing error: Image exceeds size limit and JPEG conversion failed - {jpeg_error}"
                
                else:
                    print(f"API call error, attempting retry: {api_error}")
                    # Simple retry mechanism
                    time.sleep(2)
                    try:
                        response = client.messages.create(
                            model=model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt_text},
                                        image_content
                                    ]
                                }
                            ]
                        )
                        model_response = response.content[0].text
                    except Exception as retry_error:
                        raise Exception(f"Retry failed: {retry_error}")
            
            result = {
                "Question_id": question_id,
                "Response": model_response,
                "Answer": item.get('Answer'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result)  # Add new result to the list

            # Save results every 5 *new* questions or at the end of the file
            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} questions (added {newly_processed_count} new), saved results to {output_path}.")

        except Exception as e:
            print(f"Error processing Question_id {question_id}: {e}")
            result = {
                "Question_id": question_id,
                "Response": f"Processing error: {e}",  # Record error message
                "Answer": item.get('Answer'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }
            results.append(result)

            # Save results when error occurs
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Question_id {question_id}, saved current results to {output_path}.")

    # Ensure final results are saved at the end of function if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"File processing complete, final results saved to: {output_path}")

    return results  # Return the complete results list for the current file

def main(data_folder, api_key, model_name, output_base_dir, categories=None, max_tokens=1024, temperature=0):
    """Main function to process specified parquet datasets and generate outputs in separate files."""
    # Ensure base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Find all Parquet files
    all_parquet_files = glob.glob(os.path.join(data_folder, "*.parquet"))
    files_to_process = []

    # Filter files by categories
    if categories:
        # Make sure category names don't include .parquet suffix
        category_basenames = {cat.replace(".parquet", "") for cat in categories}
        for file_path in all_parquet_files:
            basename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            if basename_no_ext in category_basenames:
                files_to_process.append(file_path)
        print(f"Will process specified category files: {files_to_process}")
        # Check for missing categories
        found_basenames = {os.path.splitext(os.path.basename(f))[0] for f in files_to_process}
        missing_categories = category_basenames - found_basenames
        if missing_categories:
            print(f"Warning: The following specified category files were not found: {missing_categories}")

    else:
        files_to_process = all_parquet_files
        print(f"Will process all .parquet files in the {data_folder} folder.")

    if not files_to_process:
        print("No Parquet files found to process.")
        return

    # Initialize Claude API client
    print("Initializing Claude API client...")
    try:
        client = Anthropic(api_key=api_key)
        print("Claude API client initialization complete.")
    except Exception as e:
        print(f"Error initializing Claude API client: {e}")
        return

    # Process each file
    for file_path in files_to_process:
        print("-" * 50)
        print(f"Starting to process file: {file_path}")

        # 1. Dynamically generate output filename from input filename
        base_name = os.path.basename(file_path)  # e.g., "Multi-window_QA.parquet"
        dataset_name = os.path.splitext(base_name)[0]  # e.g., "Multi-window_QA"
        # Use "claude_" prefix and dataset name to create JSON filename
        output_filename = f"claude_{dataset_name}.json"
        current_output_path = os.path.join(output_base_dir, output_filename)  # Full output path
        print(f"Results will be saved to: {current_output_path}")

        # 2. Load existing results for current file
        existing_results_for_current_file = load_existing_results(current_output_path)

        # 3. Load Parquet data
        data_list = load_parquet_data(file_path)

        # 4. Process data
        if data_list:
            # Pass dynamically generated output path and loaded existing results to process_data
            process_data(
                data_list,
                client,
                model_name,
                temperature,
                max_tokens,
                existing_results=existing_results_for_current_file,  # Pass existing results
                output_path=current_output_path                      # Pass output path
            )
            print(f"File {file_path} processing complete.")
        else:
            print(f"No data in file {file_path} or loading failed.")

    print("=" * 50)
    print("Processing of all specified files complete.")

if __name__ == "__main__":
    data_folder = "./datasets"  # Data folder path
    api_key = "your_key"
    model_name = "claude-3-7-sonnet-20250219"
    
    output_dir = "./results/claude"  # Specify result directory
    
    categories = ["Real-world_QA", "Synthetic_QA", "Multi-window_QA"]
    
    max_tokens = 300  # Maximum tokens to generate
    temperature = 0  # Temperature parameter
    
    # Call main function
    main(data_folder, api_key, model_name, output_dir, categories, max_tokens, temperature)