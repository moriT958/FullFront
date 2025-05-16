"""
Gemini API Visual Question Answering Tool

This script processes various QA datasets using Google's Gemini API:
- Real-world QA
- Synthetic QA
- Multi-window QA

Usage:
1. Set your API key in the main function
2. Make sure datasets are in the ./datasets folder
3. Specify which categories to process
4. Run the script: python gemini_qa.py

Results will be saved to ./results/gemini directory.
"""

import os
import json
import base64
import glob
import pandas as pd
from io import BytesIO
from google import genai
from google.genai import types
import time

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


def process_data(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    """Process data list, generate model responses, and save results in real-time."""
    results = []
    
    # Create a set of processed question IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Question_id') for item in existing_results if item.get('Question_id')}
        results = existing_results.copy() # Use existing results for this file
        print(f"Loaded {len(processed_ids)} already processed questions (from {output_path}), will skip these questions.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        question_id = item.get('Question_id')

        # Skip already processed questions
        if question_id in processed_ids:
            print(f"Skipping already processed question ID: {question_id}")
            continue

        newly_processed_count += 1 # Count newly processed questions

        try:
            # Decode base64 image
            base64_image = item['Image']
            image_bytes = base64.b64decode(base64_image)
            
            prompt = item['Prompt']
            question = item['Question']
            choices = item['Choices']
            
            # Build prompt text
            prompt_text = f"{prompt}\nQuestion: {question}\nChoices: {choices}"
            
            # Build API request
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt_text,
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    ],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        thinking_config=thinking_config,
                        max_output_tokens=max_tokens
                    )
                )
                
                # Get response text
                model_response = response.text.strip()
                
            except Exception as api_error:
                print(f"API call error, attempting retry: {api_error}")
                # Simple retry mechanism
                time.sleep(60)
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            prompt_text,
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                        ],
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            thinking_config=thinking_config,
                            max_output_tokens=max_tokens
                        )
                    )
                    model_response = response.text.strip()
                except Exception as retry_error:
                    raise Exception(f"Retry failed: {retry_error}")
            
            result = {
                "Question_id": question_id,
                "Response": model_response,
                "Answer": item.get('Answer'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result) # Add new result to list

            # Save results every 5 new questions or at the last new question
            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path) # Save to specified file
                     print(f"Processed {i + 1}/{total_items} questions (added {newly_processed_count} new), real-time saving to {output_path}.")
                     time.sleep(60)

        except Exception as e:
            print(f"Error processing Question_id {question_id}: {e}")
            result = {
                "Question_id": question_id,
                "Response": f"Processing error: {e}", # Record error message
                "Answer": item.get('Answer'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }
            results.append(result)

            # Also save results when an error occurs
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Question_id {question_id}, current results saved to {output_path}.")

    # Ensure final results are saved at the end of the function if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Finished processing file, final results saved to: {output_path}")

    return results # Return complete result list for this file

def main(data_folder, api_key, model_name, output_base_dir, categories=None, max_tokens=1024, temperature=0):
    """Main function, processes Parquet datasets in the specified folder and generates a separate output file for each dataset."""
    # Ensure the base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Find all Parquet files
    all_parquet_files = glob.glob(os.path.join(data_folder, "*.parquet"))
    files_to_process = []

    # Filter files based on categories
    if categories:
        # Ensure category names don't include .parquet suffix for matching
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
            print(f"Warning: The following specified category files were not found in the folder: {missing_categories}")

    else:
        files_to_process = all_parquet_files
        print(f"Will process all .parquet files in the folder {data_folder}.")

    if not files_to_process:
        print("No Parquet files found to process.")
        return

    # --- Initialize Gemini API client ---
    print("Initializing Gemini API client...")
    try:
        client = genai.Client(api_key=api_key)
        print("Gemini API client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini API client: {e}")
        return

    # --- Process each file ---
    for file_path in files_to_process:
        print("-" * 50)
        print(f"Starting to process file: {file_path}")

        # 1. Dynamically generate output filename from input filename
        base_name = os.path.basename(file_path) # e.g., "Multi-window_QA.parquet"
        dataset_name = os.path.splitext(base_name)[0] # e.g., "Multi-window_QA"
        # Combine "gemini_" prefix with the file name to create a JSON filename
        output_filename = f"gemini_{dataset_name}.json"
        current_output_path = os.path.join(output_base_dir, output_filename) # Full output path
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
                existing_results=existing_results_for_current_file, # Pass existing results for current file
                output_path=current_output_path                     # Pass output path for current file
            )
            print(f"File {file_path} processing completed.")
        else:
            print(f"No data in file {file_path} or loading failed.")

    print("=" * 50)
    print("All specified files processing completed.")

if __name__ == "__main__":
    data_folder = "./datasets"  # Data folder path
    api_key = "your_api_key"
    model_name = "gemini-2.5-flash-preview-04-17"  # Gemini model name
    
    output_dir = "./results/gemini"  # Directory to save results
    
    # Specify which dataset categories to process (just provide base names like "Multi-window_QA")
    categories = ["Real-world_QA", "Synthetic_QA", "Multi-window_QA"]
    
    max_tokens = 300  # Maximum tokens to generate
    temperature = 0  # Temperature parameter
    thinking_config = types.ThinkingConfig(thinking_budget=0)
    
    # Call main function
    main(data_folder, api_key, model_name, output_dir, categories, max_tokens, temperature)