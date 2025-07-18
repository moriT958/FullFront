"""
HTML to Image Renderer

Usage:
1. Run the script directly: python render_img.py
2. Or import the capture_screenshot function:
   from render_img import capture_screenshot
   capture_screenshot(html_file_path, screenshot_folder)

This script renders HTML files to PNG images using Playwright.
It waits for all images to load, handles lazy-loaded content, and
captures full-page screenshots.
"""

import os
import time

from playwright.sync_api import sync_playwright


def capture_screenshot(html_file_path, screenshot_folder):
    """
    Capture a screenshot of an HTML file after all images are loaded.

    Args:
        html_file_path: Path to the HTML file
        screenshot_folder: Path to save screenshots

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure HTML file exists
        if not os.path.exists(html_file_path):
            raise FileNotFoundError(f"HTML file not found: {html_file_path}")

        # Get absolute path
        absolute_path = os.path.abspath(html_file_path)
        file_url = f"file://{absolute_path}"

        # Generate screenshot filename
        html_file_name = os.path.splitext(os.path.basename(html_file_path))[0]
        screenshot_path = os.path.join(screenshot_folder, f"{html_file_name}.png")

        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to HTML file
            page.goto(file_url)

            # Wait for page to load
            page.wait_for_load_state("load")

            # Wait for all images to fully load
            page.evaluate(
                """
            () => {
                return new Promise((resolve) => {
                    const images = document.querySelectorAll('img');
                    let loadedImages = 0;
                    
                    // If no images, resolve immediately
                    if (images.length === 0) {
                        return resolve();
                    }
                    
                    // Add load event handlers to each image
                    images.forEach(img => {
                        // Already loaded images
                        if (img.complete) {
                            loadedImages++;
                            if (loadedImages === images.length) {
                                resolve();
                            }
                        } else {
                            // Listen for load events
                            img.addEventListener('load', () => {
                                loadedImages++;
                                if (loadedImages === images.length) {
                                    resolve();
                                }
                            });
                            
                            // Listen for error events
                            img.addEventListener('error', () => {
                                loadedImages++;
                                if (loadedImages === images.length) {
                                    resolve();
                                }
                            });
                        }
                    });
                });
            }
            """
            )

            # Simulate scrolling to load lazy-loaded elements
            page.evaluate(
                """
            () => {
                return new Promise((resolve) => {
                    let totalHeight = 0;
                    const distance = 100;
                    const timer = setInterval(() => {
                        const scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;
                        
                        // Stop if reached bottom or scrolled enough
                        if(totalHeight >= scrollHeight || totalHeight > 10000){
                            clearInterval(timer);
                            // Scroll back to top
                            window.scrollTo(0, 0);
                            setTimeout(resolve, 100); // Give time for page to stabilize
                        }
                    }, 100);
                });
            }
            """
            )

            # Capture screenshot
            page.screenshot(path=screenshot_path, full_page=True)

            # Close browser
            browser.close()

            print(f"Screenshot saved to: {screenshot_path}")
            return True
    except Exception as e:
        print(f"Error processing file {html_file_path}: {str(e)}")
        return False


if __name__ == "__main__":
    start_time = time.time()
    # HTML files directory
    html_folder = "generated_htmls"  # Replace with your HTML files directory
    # Screenshots output directory
    screenshot_folder = (
        "pro_imgs/openai/Image_to_code"  # Replace with your screenshots directory
    )

    # Ensure screenshot folder exists
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    # Get all HTML files
    html_files = [f for f in os.listdir(html_folder) if f.endswith(".html")]

    # Sort alphabetically
    html_files.sort()

    # Get existing screenshot names (without extension)
    existing_screenshots = set()
    if os.path.exists(screenshot_folder):
        for file in os.listdir(screenshot_folder):
            if file.endswith(".png"):
                existing_screenshots.add(os.path.splitext(file)[0])

    # Track failed files
    failed_indices = []
    # Count skipped files
    skipped_count = 0

    # Process each HTML file
    for index, html_file in enumerate(html_files):
        # Get HTML filename without extension
        html_file_name = os.path.splitext(html_file)[0]

        # Skip if screenshot already exists
        if html_file_name in existing_screenshots:
            print(
                f"Screenshot exists, skipping [{index+1}/{len(html_files)}]: {html_file}"
            )
            skipped_count += 1
            continue

        print(f"Processing [{index+1}/{len(html_files)}]: {html_file}")
        html_file_path = os.path.join(html_folder, html_file)
        success = capture_screenshot(html_file_path, screenshot_folder)
        if not success:
            failed_indices.append(index)

    # Print processing statistics
    print("\nProcessing Statistics:")
    print(f"Total files: {len(html_files)}")
    print(f"Skipped: {skipped_count}")
    print(f"Newly processed: {len(html_files) - skipped_count - len(failed_indices)}")

    # Print failed files
    if failed_indices:
        print("\nFailed files indices:")
        for idx in failed_indices:
            print(f"Index {idx}: {html_files[idx]}")
        print(f"Failed count: {len(failed_indices)}")
    else:
        print("\nAll files processed successfully!")
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time} seconds")

