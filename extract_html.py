"""
Script to extract HTML content from JSON files in results directory and save as HTML files.
"""

import json
import os
from pathlib import Path


def extract_html_from_results(json_filename, output_dir):
    """Extract HTML content from JSON results and save as individual HTML files."""

    # Input and output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Processing JSON files from: {json_filename}")
    print(f"Output directory: {output_dir}")

    fp_json = open(json_filename, "r")
    json_data = json.load(fp_json)

    for v in json_data:
        p = os.path.join(str(output_dir), v["Png_id"] + ".html")
        fp_html = open(p, "w")
        fp_html.write(str(v["Response"]))
        fp_html.close()

    fp_json.close()


if __name__ == "__main__":
    source_dir = "./results/openai/openai_Image_to_code.json"
    output_dir = "./generated_htmls"
    extract_html_from_results(source_dir, output_dir)
