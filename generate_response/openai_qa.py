"""
OpenAI Vision API Question Answering Script

Usage:
1. Place your dataset parquet files in the './datasets' folder
2. Configure your OpenAI API key in the main function
3. Set the desired model name (default: gpt-4o-2024-11-20)
4. Run the script: python openai_qa.py
5. Results will be saved to './results/openai' directory

The script processes parquet files containing images (base64 encoded) and questions,
sends them to OpenAI vision API, and saves the responses as JSON files.
"""

import os
import json
import glob
import pandas as pd
from openai import OpenAI
import time

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
            print(f"Failed to load existing results from {output_path}: {e}")
    return []

def save_results_to_json(results, output_path="output.json"):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save results to {output_path}: {e}")


def process_data(data_list, client, model_name, temperature, max_tokens, existing_results=None, output_path=None):
    results = []
    
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Question_id') for item in existing_results if item.get('Question_id')}
        results = existing_results.copy() 
        print(f"Loaded {len(processed_ids)} previously processed questions (from {output_path}), will skip them.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        question_id = item.get('Question_id')

        if question_id in processed_ids:
            print(f"Skipping already processed question ID: {question_id}")
            continue

        newly_processed_count += 1

        try:
            base64_image = item['Image']
            
            prompt = item['Prompt']
            question = item['Question']
            choices = item['Choices']
            
            prompt_text = f"{prompt}\nQuestion: {question}\nChoices: {choices}"
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
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
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                model_response = response.choices[0].message.content
                
            except Exception as api_error:
                print(f"API call error, attempting retry: {api_error}")
                time.sleep(2)
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
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
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    model_response = response.choices[0].message.content
                except Exception as retry_error:
                    raise Exception(f"Retry failed: {retry_error}")
            
            result = {
                "Question_id": question_id,
                "Response": model_response,
                "Answer": item.get('Answer'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result)

            # Save results every 5 new questions or at the last item
            if newly_processed_count > 0 and (newly_processed_count % 5 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} questions (added {newly_processed_count} new), results saved to {output_path}.")

        except Exception as e:
            print(f"Error processing Question_id {question_id}: {e}")
            result = {
                "Question_id": question_id,
                "Response": f"Processing error: {e}",
                "Answer": item.get('Answer'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }
            results.append(result)

            # Save results when error occurs
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Question_id {question_id}, current results saved to {output_path}.")

    # Ensure final results are saved at the end if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"File processing complete, final results saved to: {output_path}")

    return results

def main(data_folder, api_key, model_name, output_base_dir, categories=None, max_tokens=1024, temperature=0):
    """Main function to process parquet datasets and generate output files for each dataset."""
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all parquet files
    all_parquet_files = glob.glob(os.path.join(data_folder, "*.parquet"))
    files_to_process = []

    # Filter files by categories if specified
    if categories:
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
            print(f"Warning: The following specified categories were not found: {missing_categories}")
    else:
        files_to_process = all_parquet_files
        print(f"Will process all .parquet files in the {data_folder} folder.")

    if not files_to_process:
        print("No parquet files found to process.")
        return

    # Initialize OpenAI API client
    print("Initializing OpenAI API client...")
    try:
        client = OpenAI(api_key=api_key)
        print("OpenAI API client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI API client: {e}")
        return

    # Process each file
    for file_path in files_to_process:
        print("-" * 50)
        print(f"Starting to process file: {file_path}")

        # Generate output filename from input filename
        base_name = os.path.basename(file_path) 
        dataset_name = os.path.splitext(base_name)[0] 
        output_filename = f"openai_{dataset_name}.json"
        current_output_path = os.path.join(output_base_dir, output_filename)
        print(f"Results will be saved to: {current_output_path}")

        # Load existing results for the current file
        existing_results_for_current_file = load_existing_results(current_output_path)

        # Load parquet data
        data_list = load_parquet_data(file_path)

        # Process data
        if data_list:
            process_data(
                data_list,
                client,
                model_name,
                temperature,
                max_tokens,
                existing_results=existing_results_for_current_file,
                output_path=current_output_path
            )
            print(f"File {file_path} processing completed.")
        else:
            print(f"No data found or failed to load file {file_path}.")

    print("=" * 50)
    print("All specified files have been processed.")

if __name__ == "__main__":
    data_folder = "./datasets"  
    api_key = "your_api_key"  
    model_name = "gpt-4o-2024-11-20"  
    
    output_dir = "./results/openai"
    
    categories = ["Real-world_QA", "Synthetic_QA", "Multi-window_QA"]
    
    max_tokens = 300
    temperature = 0 
    
    main(data_folder, api_key, model_name, output_dir, categories, max_tokens, temperature)