"""
OpenAI UI Dataset Processing Tool

Usage:
1. Install requirements: pip install pandas openai pillow
2. Set your OpenAI API key in the api_key variable
3. Run the script: python openai_code.py
4. Results will be saved to ./results/openai/

You can modify the categories, model_name, and other parameters in the main section.
"""

import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# Utility functions
def load_parquet_data(file_path):
    """Load data from a single Parquet file."""
    try:
        df = pd.read_parquet(file_path)
        return df.to_dict("records")
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return []


def load_existing_results(output_path):
    """Load existing results to skip already processed data."""
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load existing results file {output_path}: {e}")
    return []


def save_results_to_json(results, output_path="output.json"):
    """Save results to a JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save results to {output_path}: {e}")


def extract_html_code(text):
    """Extract HTML code from text, requires it starts with <!DOCTYPE html> and ends with </html>.
    Returns the original text if no matching HTML code is found."""
    html_pattern = r"(?:<\!DOCTYPE\s+html>.*?<\/html>)"
    matches = re.findall(html_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0]
    if "<!DOCTYPE html>" in text.lower() and "</html>" in text.lower():
        start = text.lower().find("<!doctype html>")
        end = text.lower().find("</html>") + len("</html>")
        return text[start:end]
    return text


def decode_base64_image(base64_string):
    """Validate base64 encoded image string."""
    try:
        base64.b64decode(base64_string)
        return base64_string  # For OpenAI, we return the original base64 string
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None


def generate_openai_response(
    client, model_name, messages, temperature, max_tokens, retries=2
):
    """Call OpenAI API to generate a response, with retry mechanism."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"API call failed, retrying in {wait_time} seconds: {e}")
                time.sleep(wait_time)
            else:
                raise Exception(f"API call failed after {retries+1} attempts: {e}")


def process_text_to_code(
    data_list,
    client,
    model_name,
    temperature,
    max_tokens,
    existing_results=None,
    output_path=None,
):
    """Process text to code conversion data."""
    results = []

    # Create a set of processed question IDs to skip
    processed_ids = set()
    if existing_results:
        processed_ids = {item.get("Id") for item in existing_results if item.get("Id")}
        results = existing_results.copy()
        print(
            f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items."
        )

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get("Id")

        # Skip already processed questions
        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            prompt = item["Prompt"]
            input_text = item["Input_text"]

            # Build prompt text
            prompt_text = f"{prompt}\nDescription:{input_text}"

            # Build API request messages
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
            ]

            # Generate response
            original_response = generate_openai_response(
                client,
                model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract HTML code
            html_response = extract_html_code(original_response)

            # Log if HTML code was successfully extracted
            if html_response != original_response:
                print(f"ID {item_id}: Successfully extracted HTML code")

            result = {
                "Id": item_id,
                "Response": html_response,
                "Label_html": item.get("Label_html"),
                "Category": item.get("Category"),
                "Png_id": item.get("Png_id"),
            }

            results.append(result)

            # Save results every 5 new items or at the last item
            if newly_processed_count > 0 and (
                newly_processed_count % 5 == 0 or i == total_items - 1
            ):
                if output_path:
                    save_results_to_json(results, output_path)
                    print(
                        f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saving results to {output_path}."
                    )

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Response": f"Processing error: {e}",
                "Label_html": item.get("Label_html"),
                "Category": item.get("Category"),
                "Png_id": item.get("Png_id"),
            }
            results.append(result)

            # Save results on error
            if output_path:
                save_results_to_json(results, output_path)
                print(
                    f"Error processing Id {item_id}, saved current results to {output_path}."
                )

    # Ensure final results are saved if any new items were processed
    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Finished processing file, final results saved to: {output_path}")

    return results


def process_image_to_code(
    data_list,
    client,
    model_name,
    temperature,
    max_tokens,
    existing_results=None,
    output_path=None,
):
    """Process image to code conversion data."""
    results = []

    processed_ids = set()
    if existing_results:
        processed_ids = {item.get("Id") for item in existing_results if item.get("Id")}
        results = existing_results.copy()
        print(
            f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items."
        )

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get("Id")

        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            # Get base64 image
            base64_image = item["Image"]["bytes"]
            if not base64_image:
                raise ValueError("Image data is empty")
            base64_string = base64.b64encode(base64_image).decode("utf-8")

            prompt = item["Prompt"]

            # Build API request messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_string}"
                            },
                        },
                    ],
                }
            ]

            # Generate response
            original_response = generate_openai_response(
                client,
                model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract HTML code
            html_response = extract_html_code(original_response)

            if html_response != original_response:
                print(f"ID {item_id}: Successfully extracted HTML code")

            result = {
                "Id": item_id,
                "Response": html_response,
                "Label_html": item.get("Label_html"),
                "Category": item.get("Category"),
                "Png_id": item.get("Png_id"),
            }

            results.append(result)

            if newly_processed_count > 0 and (
                newly_processed_count % 5 == 0 or i == total_items - 1
            ):
                if output_path:
                    save_results_to_json(results, output_path)
                    print(
                        f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saving results to {output_path}."
                    )

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Response": f"Processing error: {e}",
                "Label_html": item.get("Label_html"),
                "Category": item.get("Category"),
                "Png_id": item.get("Png_id"),
            }
            results.append(result)

            if output_path:
                save_results_to_json(results, output_path)
                print(
                    f"Error processing Id {item_id}, saved current results to {output_path}."
                )

    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Finished processing file, final results saved to: {output_path}")

    return results


def process_refinement_to_code(
    data_list,
    client,
    model_name,
    temperature,
    max_tokens,
    existing_results=None,
    output_path=None,
):
    results = []

    processed_ids = set()
    if existing_results:
        processed_ids = {item.get("Id") for item in existing_results if item.get("Id")}
        results = existing_results.copy()
        print(
            f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items."
        )

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get("Id")

        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            # Get base64 image
            base64_image = item["Image"]
            if not base64_image:
                raise ValueError("Image data is empty")

            prompt = item["Prompt"]
            input_html = item["Input_html"]

            # Build prompt text
            prompt_text = f"{prompt}\nCode:\n{input_html}"

            # Build API request messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_string}"
                            },
                        },
                    ],
                }
            ]

            # Generate response
            original_response = generate_openai_response(
                client,
                model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract HTML code
            html_response = extract_html_code(original_response)

            if html_response != original_response:
                print(f"ID {item_id}: Successfully extracted HTML code")

            result = {
                "Id": item_id,
                "Response": html_response,
                "Label_html": item.get("Label_html"),
                "Category": item.get("Category"),
                "Png_id": item.get("Png_id"),
            }

            results.append(result)

            if newly_processed_count > 0 and (
                newly_processed_count % 5 == 0 or i == total_items - 1
            ):
                if output_path:
                    save_results_to_json(results, output_path)
                    print(
                        f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saving results to {output_path}."
                    )

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Response": f"Processing error: {e}",
                "Label_html": item.get("Label_html"),
                "Category": item.get("Category"),
                "Png_id": item.get("Png_id"),
            }
            results.append(result)

            if output_path:
                save_results_to_json(results, output_path)
                print(
                    f"Error processing Id {item_id}, saved current results to {output_path}."
                )

    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Finished processing file, final results saved to: {output_path}")

    return results


def process_interaction_to_code(
    data_list,
    client,
    model_name,
    temperature,
    max_tokens,
    existing_results=None,
    output_path=None,
):
    """Process interaction to code conversion data."""
    results = []

    processed_ids = set()
    if existing_results:
        processed_ids = {item.get("Id") for item in existing_results if item.get("Id")}
        results = existing_results.copy()
        print(
            f"Loaded {len(processed_ids)} already processed items (from {output_path}), will skip these items."
        )

    total_items = len(data_list)
    newly_processed_count = 0

    for i, item in enumerate(data_list):
        item_id = item.get("Id")

        if item_id in processed_ids:
            continue

        newly_processed_count += 1

        try:
            # Get two base64 images
            before_image = item["Before_image"]
            after_image = item["After_image"]

            # Check if images are valid
            if not before_image or not after_image:
                raise ValueError("Image data is empty")

            prompt = item["Prompt"]

            # Build API request messages - using two images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{before_image}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{after_image}"
                            },
                        },
                    ],
                }
            ]

            # Generate response
            original_response = generate_openai_response(
                client,
                model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract HTML code
            html_response = extract_html_code(original_response)

            if html_response != original_response:
                print(f"ID {item_id}: Successfully extracted HTML code")

            result = {
                "Id": item_id,
                "Interaction_type": item.get("Interaction_type"),
                "Response": html_response,
                "Label_html": item.get("Label_html"),
                "Category": item.get("Category"),
                "Png_id": item.get("Png_id"),
            }

            results.append(result)

            if newly_processed_count > 0 and (
                newly_processed_count % 5 == 0 or i == total_items - 1
            ):
                if output_path:
                    save_results_to_json(results, output_path)
                    print(
                        f"Processed {i + 1}/{total_items} items (added {newly_processed_count} new), saving results to {output_path}."
                    )

        except Exception as e:
            print(f"Error processing Id {item_id}: {e}")
            result = {
                "Id": item_id,
                "Interaction_type": item.get("Interaction_type"),
                "Response": f"Processing error: {e}",
                "Label_html": item.get("Label_html"),
                "Category": item.get("Category"),
                "Png_id": item.get("Png_id"),
            }
            results.append(result)

            if output_path:
                save_results_to_json(results, output_path)
                print(
                    f"Error processing Id {item_id}, saved current results to {output_path}."
                )

    if newly_processed_count > 0 and output_path:
        save_results_to_json(results, output_path)
        print(f"Finished processing file, final results saved to: {output_path}")

    return results


def main(
    data_folder,
    api_key,
    model_name,
    output_base_dir,
    categories=None,
    max_tokens=8000,
    temperature=0,
):
    """Main function, selects processing method based on specified categories."""
    # Ensure base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # All supported task types and corresponding processing functions
    task_handlers = {
        "Text_to_code": process_text_to_code,
        "Image_to_code": process_image_to_code,
        "Code_Refinement": process_refinement_to_code,
        "Interaction_Authoring": process_interaction_to_code,
    }

    # If no categories specified, default to all supported categories
    if categories is None:
        categories = list(task_handlers.keys())

    # Validate that all specified categories are supported
    unsupported_categories = [cat for cat in categories if cat not in task_handlers]
    if unsupported_categories:
        print(
            f"Warning: The following categories are not supported: {unsupported_categories}"
        )
        # Filter out unsupported categories
        categories = [cat for cat in categories if cat in task_handlers]
        if not categories:
            print("No valid categories to process.")
            return

    print(f"Will process the following categories: {categories}")

    # --- Initialize OpenAI API client ---
    print("Initializing OpenAI API client...")
    try:
        # Instantiate OpenAI API client
        client = OpenAI(api_key=api_key)
        print("OpenAI API client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI API client: {e}")
        return

    # --- Process each selected category ---
    for category in categories:
        print("=" * 50)
        print(f"Starting to process category: {category}")

        # Find corresponding parquet file for the category
        target_file = os.path.join(data_folder, f"{category}.parquet")

        if not os.path.exists(target_file):
            print(f"Cannot find file for category {category}: {target_file}")
            continue

        print(f"Found category file: {target_file}")

        # Generate output filename
        output_filename = f"openai_{category}.json"
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
            output_path=current_output_path,
        )

        print(f"Category {category} processing completed.")

    print("=" * 50)
    print("All specified category processing workflows completed.")


if __name__ == "__main__":
    load_dotenv(".env")

    data_folder = "./datasets"
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    output_dir = "./results/openai"

    # Specify categories to process
    categories = [
        # "Text_to_code",
        "Image_to_code",
        # "Code_Refinement",
        # "Interaction_Authoring",
    ]

    max_tokens = 16000
    temperature = 0

    # Call main function
    main(
        data_folder,
        api_key,
        model_name,
        output_dir,
        categories,
        max_tokens,
        temperature,
    )
