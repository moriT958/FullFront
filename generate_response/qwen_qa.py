"""
Qwen Vision-Language Model Question Answering Script

Usage:
    python qwen_qa.py [--data_folder DATA_FOLDER] [--model_path MODEL_PATH] 
                      [--output_dir OUTPUT_DIR] [--categories CATEGORIES] 
                      [--max_tokens MAX_TOKENS] [--temperature TEMPERATURE]
                      [--tensor_parallel_size TENSOR_PARALLEL_SIZE]

This script processes image-based questions from parquet files, generates responses
using Qwen Vision-Language model, and saves results to JSON files.
"""

import os
import json
import base64
import glob
import pandas as pd
from PIL import Image
from io import BytesIO
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# --- load_parquet_data, load_existing_results, save_results_to_json, process_data 函数保持不变 ---

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


def process_data(data_list, llm, processor, temperature, max_tokens, existing_results=None, output_path=None):
    """Process data list, generate model responses, and save results in real-time."""
    results = []
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Create a set of processed question IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get('Question_id') for item in existing_results if item.get('Question_id')}
        results = existing_results.copy()  # Use existing results for this file
        print(f"Loaded {len(processed_ids)} already processed questions (from {output_path}), will skip them.")

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        question_id = item.get('Question_id')

        # Skip already processed questions
        if question_id in processed_ids:
            continue

        newly_processed_count += 1  # Count newly processed questions

        try:
            # Decode base64 image
            base64_image = item['Image']
            image_bytes = base64.b64decode(base64_image)
            image_io = BytesIO(image_bytes)
            # Convert to PIL Image object
            image = Image.open(image_io)

            prompt = item['Prompt']
            question = item['Question']
            choices = item['Choices']

            # Build messages
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{prompt}\n'Question:'{question}\n'Choices:'+{choices}"}
                ]}
            ]

            # Process messages
            chat_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            llm_inputs = {
                "prompt": chat_prompt,
                "multi_modal_data": mm_data,
            }

            # Generate response
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
            response = outputs[0].outputs[0].text.strip()

            result = {
                "Question_id": question_id,
                "Response": response,
                "Answer": item.get('Answer'),
                "Category": item.get('Category'),
                "Png_id": item.get('Png_id')
            }

            results.append(result)  # Add new result to the list

            # Save results every 10 *new* questions or at the last *new* question
            if newly_processed_count > 0 and (newly_processed_count % 10 == 0 or i == total_items - 1):
                 if output_path:
                     save_results_to_json(results, output_path)
                     print(f"Processed {i + 1}/{total_items} questions (added {newly_processed_count} new), saving results to {output_path}.")

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

            # Save results when an error occurs
            if output_path:
                save_results_to_json(results, output_path)
                print(f"Error processing Question_id {question_id}, current results saved to {output_path}.")

    # Ensure final results are saved at the end of function if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"File processing complete, final results saved to: {output_path}")

    return results  # Return the complete list of results for this file

def main(data_folder, model_path, output_base_dir, categories=None, max_tokens=256, temperature=0.1, tensor_parallel_size=8):
    """Main function to process specified parquet datasets and generate responses, with separate output files for each dataset."""
    # Ensure base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Find all Parquet files
    all_parquet_files = glob.glob(os.path.join(data_folder, "*.parquet"))
    files_to_process = []

    # Filter files by categories
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
        print(f"Will process all .parquet files in folder {data_folder}.")

    if not files_to_process:
        print("No Parquet files found to process.")
        return

    # Load model
    print("Loading model...")
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 1, "video": 0},
            enforce_eager=True
        )
        processor = AutoProcessor.from_pretrained(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process each file
    for file_path in files_to_process:
        print("-" * 50)
        print(f"Starting to process file: {file_path}")

        # 1. Dynamically generate output filename from input filename
        base_name = os.path.basename(file_path)  # e.g., "Multi-window_QA.parquet"
        dataset_name = os.path.splitext(base_name)[0]  # e.g., "Multi-window_QA"
        # Combine "qwen_" prefix with filename to make a JSON filename
        output_filename = f"qwen_{dataset_name}.json"
        current_output_path = os.path.join(output_base_dir, output_filename)  # Complete output path
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
                llm,
                processor,
                temperature,
                max_tokens,
                existing_results=existing_results_for_current_file,  # Pass existing results for current file
                output_path=current_output_path                      # Pass output path for current file
            )
            print(f"File {file_path} processing complete.")
        else:
            print(f"No data in file {file_path} or file loading failed.")

    print("=" * 50)
    print("Processing complete for all specified files.")

if __name__ == "__main__":
    # Use relative paths
    data_folder = "data/datasets"  # Data folder path
    model_path = "models/Qwen2.5-VL-72B-Instruct"  # Model weights path
    output_dir = "results"  # Output directory for results
    
    # Specify categories to process (only provide base names, e.g., "Multi-window_QA")
    categories = ["Real-world_QA", "Synthetic_QA", "Multi-window_QA"]
    # categories = None  # Set to None to process all parquet files
    
    max_tokens = 1024  # Maximum tokens to generate
    temperature = 0  # Temperature parameter
    tensor_parallel_size = 8  # Use 8 GPUs in parallel
    
    # Call main function, passing the output directory
    main(data_folder, model_path, output_dir, categories, max_tokens, temperature, tensor_parallel_size)