"""
Script to extract images from datasets/Image_to_code.parquet and save them to pro_imgs folder structure.
"""

import io
from pathlib import Path

import pandas as pd
from PIL import Image


def extract_images_from_parquet(datasets):
    """Extract images from parquet file and save to pro_imgs structure."""

    # Read the parquet file
    print("Reading parquet file...")
    df = pd.read_parquet(datasets)

    # Create base directory structure
    base_dir = Path("label_imgs")
    image_to_code_dir = base_dir / "Image_to_code"
    image_to_code_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(df)} records in parquet file")
    print(f"Columns: {df.columns.tolist()}")

    # Extract and save images
    saved_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        try:
            # Get image data from the 'Image' column
            if "Image" in row and row["Image"] is not None:
                image_data = row["Image"]

                # Handle different image data formats
                if isinstance(image_data, dict) and "bytes" in image_data:
                    # Image data is in dict format with 'bytes' key
                    image_bytes = image_data["bytes"]
                elif isinstance(image_data, bytes):
                    # Image data is directly bytes
                    image_bytes = image_data
                else:
                    print(
                        f"Warning: Unexpected image data format for row {idx}: {type(image_data)}"
                    )
                    continue

                # Get filename - use Png_id if available, otherwise use Id
                if "Png_id" in row and row["Png_id"]:
                    filename = f"{row['Png_id']}.png"
                elif "Id" in row and row["Id"]:
                    filename = f"img_{row['Id']}.png"
                else:
                    filename = f"img_{idx}.png"

                # Save image
                image_path = image_to_code_dir / filename

                # Convert bytes to PIL Image and save
                image = Image.open(io.BytesIO(image_bytes))
                image.save(image_path, "PNG")

                saved_count += 1
                if saved_count % 10 == 0:
                    print(f"Saved {saved_count} images...")

            else:
                print(f"No image data found for row {idx}")
                error_count += 1

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            error_count += 1

    print(f"\nCompleted!")
    print(f"Successfully saved: {saved_count} images")
    print(f"Errors encountered: {error_count} records")
    print(f"Images saved to: {image_to_code_dir}")


if __name__ == "__main__":
    datasets_dir = "datasets/Image_to_code.parquet"
    extract_images_from_parquet(datasets_dir)
