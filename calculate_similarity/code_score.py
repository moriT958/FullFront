"""
HTML Similarity Scorer

This module calculates similarity scores between generated HTML and reference HTML.

Usage:
    - Run directly: python code_score.py
    - Import and use: 
        from code_score import HTMLComparer
        comparer = HTMLComparer()
        results = comparer.compare_html_files_content(generated_html, reference_html)

Configuration:
    - Set model_names and dataset_names in the main function
    - Input and output directories can be configured with relative paths
"""

import os
import re
import json
import numpy as np
from bs4 import BeautifulSoup, Tag
from collections import Counter
from difflib import SequenceMatcher
import pandas as pd
import urllib.parse
import cssutils
import logging
import glob

# Process JSON files and save results
def process_json_files(input_dir, output_dir, model_names, dataset_names):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    comparer = HTMLComparer()
    
    for model_name in model_names:
        for dataset_name in dataset_names:

            file_pattern = f"{model_name}_{dataset_name}.json"
            json_file = os.path.join(input_dir, file_pattern)
            
            if not os.path.exists(json_file):
                print(f"File does not exist: {file_pattern}")
                continue
            
            output_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_result.json")
            
            existing_results = []
            processed_ids = set()
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                        for item in existing_results:
                            if item['Id'] != 'summary':
                                processed_ids.add(item['Id'])
                    print(f"Loaded existing results file, skipping {len(processed_ids)} processed items")
                except Exception as e:
                    print(f"Error reading existing results file: {e}")
                    existing_results = []
            
            print(f"Processing: {file_pattern}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading file {file_pattern}: {e}")
                continue
            
            # Create result list including existing results (except summary)
            result_data = [item for item in existing_results if item['Id'] != 'summary']
            
            # Accumulated data for all results, used to calculate averages
            accumulated_results = {
                'structure_similarity': [],
                'text_content_similarity': [],
                'text_style_similarity': [],
                'image_similarity': [],
                'form_similarity': [],
                'text_implementation_rate': [],
                'image_implementation_rate': [],
                'form_implementation_rate': [],
                'total_similarity': []
            }
            
            # Collect data from existing results
            for item in result_data:
                results = item.get('Results', {})
                for key in accumulated_results:
                    if key in results:
                        accumulated_results[key].append(results[key])
            
            # Process each item
            items_processed = 0
            for item in data:
                # Extract ID
                item_id = item.get('Id')
                
                # Skip if already processed
                if item_id in processed_ids:
                    continue
                
                # Extract Response, Label_html, etc.
                response_html = item.get('Response')
                label_html = item.get('Label_html')
                category = item.get('Category')
                png_id = item.get('Png_id')
                
                if not response_html or not label_html:
                    print(f"Warning: Item {item_id} missing Response or Label_html")
                    continue
        
                try:
                    results = comparer.compare_html_files_content(response_html, label_html)
                except Exception as e:
                    print(f"Error comparing HTML (ID: {item_id}): {e}")
                    continue
                
                for key in accumulated_results:
                    if key in results:
                        accumulated_results[key].append(results[key])
                
                result_item = {
                    'Id': item_id,
                    'Category': category,
                    'Png_id': png_id,
                    'Results': results
                }
                
                result_data.append(result_item)
                items_processed += 1
                
                if items_processed % 10 == 0:
                    summary_results = {}
                    for key, values in accumulated_results.items():
                        if values:
                            summary_results[key] = sum(values) / len(values)
                        else:
                            summary_results[key] = 0
                    
                    full_result = result_data.copy()
                    full_result.append({
                        'Id': 'summary',
                        'Results': summary_results
                    })
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(full_result, f, indent=2, ensure_ascii=False)
            
            summary_results = {}
            for key, values in accumulated_results.items():
                if values:
                    summary_results[key] = sum(values) / len(values)
                else:
                    summary_results[key] = 0
            
            summary_item = {
                'Id': 'summary',
                'Results': summary_results
            }
            
            full_result = result_data.copy()
            full_result.append(summary_item)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_result, f, indent=2, ensure_ascii=False)
            
            print(f"Done: {file_pattern} -> {os.path.basename(output_file)}, processed {items_processed} items")

cssutils.log.setLevel(logging.CRITICAL)

class HTMLComparer:
    def __init__(self):
        self.form_elements = ['form', 'input', 'button', 'select', 'textarea', 'option', 'label','fieldset', 'legend', 'datalist', 'output', 'checkbox', 'radio', 'file', 'submit', 'reset', 'optgroup', 'meter', 'progress', 'keygen', 'details', 'summary', 'search', 'iframe', 'switch', 'range', 'colorpicker']
        self.text_elements = ['p', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'li', 'td', 'th', 'figcaption','div', 'pre', 'code', 'blockquote', 'cite', 'q', 'abbr', 'acronym', 'strong', 'em', 'mark', 'small', 'sub', 'sup', 'article', 'section', 'aside', 'nav', 'header', 'footer', 'main', 'figure', 'time', 'address', 'caption', 'dd', 'dl', 'dt', 'i', 'b', 'u', 'ruby', 'rt', 'rp', 'var', 'title', 'legend']
        self.image_elements = ['img', 'picture', 'source', 'map', 'area']
        
        self.style_weights = {
            'text': {
                'color': 0.15,   
                'font-size': 0.12,
                'font-weight': 0.12,
                'font-family': 0.08,
                'text-align': 0.08,
                'line-height': 0.06,
                'background-color': 0.08,
                'padding': 0.06,
                'margin': 0.06,
                'text-decoration': 0.04,
                'letter-spacing': 0.03,
                'text-shadow': 0.03,
                'text-transform': 0.03,
                'white-space': 0.03,
                'word-spacing': 0.03
            },
            'image': {
                'width': 0.25,
                'height': 0.25,
                'border-radius': 0.08,
                'margin': 0.08,
                'padding': 0.08,
                'object-fit': 0.08,
                'filter': 0.05,
                'opacity': 0.05,
                'box-shadow': 0.04,
                'aspect-ratio': 0.04
            },
            'form': {
                'width': 0.12,
                'height': 0.12,
                'background-color': 0.08,
                'border': 0.08,
                'padding': 0.08,
                'margin': 0.08,
                'color': 0.08,
                'font-size': 0.08,
                'border-radius': 0.08,
                'box-shadow': 0.05,
                'outline': 0.05,
                'transition': 0.05,
                'cursor': 0.03,
                'placeholder-style': 0.02
            }
        }
        
        self.result_weights = {
            'structure_similarity': 0.25,
            'text_content_similarity': 0.2,
            'text_style_similarity': 0.1,
            'image_similarity': 0.2, 
            'form_similarity': 0.25,
            'text_implementation_rate': 0,
            'image_implementation_rate': 0,
            'form_implementation_rate': 0
        }
    
    def similar(self, a, b):
        """Calculate similarity ratio between two strings"""
        if a is None or b is None:
            return 0
        return SequenceMatcher(None, str(a), str(b)).ratio()
    
    def extract_structure(self, soup):
        """Extract HTML structure as a tree"""
        def extract_node(node):
            if isinstance(node, Tag):
                children = [extract_node(child) for child in node.contents if isinstance(child, Tag)]
                return {'tag': node.name, 'children': children}
            return None
        
        return extract_node(soup)
    
    def structure_to_sequence(self, node):
        """Convert structure tree to sequence"""
        if not node:
            return []
        
        result = [node['tag']]
        for child in node.get('children', []):
            result.extend(self.structure_to_sequence(child))
        
        return result

    def lcs_length_with_threshold(self, X, Y, threshold=0.9):
        """Calculate longest common subsequence length with similarity threshold"""
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if self.is_similar(X[i-1], Y[j-1], threshold):
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        
        return L[m][n]

    def is_similar(self, x, y, threshold):
        """Check if two values are similar based on threshold"""
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if x == 0 and y == 0:
                return True
            max_val = max(abs(x), abs(y))
            if max_val == 0:
                return True
            similarity = 1 - abs(x - y) / max_val
            return similarity >= threshold
        elif isinstance(x, str) and isinstance(y, str):
            if not x and not y: 
                return True
            if not x or not y: 
                return False
            same_chars = sum(1 for a, b in zip(x, y) if a == b)
            max_len = max(len(x), len(y))
            similarity = same_chars / max_len
            return similarity >= threshold
        else:
            return x == y
    
    def calculate_structure_similarity(self, sequence1, sequence2):
        """Calculate structural similarity between two sequences"""
        lcs = self.lcs_length_with_threshold(sequence1, sequence2)
        return lcs / len(sequence2) if len(sequence2) > 0 else 1.0
    
    def extract_computed_styles(self, html_content):
        """Extract computed styles from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        style_map = {}
        
        # Extract inline styles
        for tag in soup.find_all(True):
            if tag.has_attr('style'):
                style_text = tag['style']
                style_dict = {}
                
                try:
                    style = cssutils.parseStyle(style_text)
                    for prop in style:
                        style_dict[prop.name] = prop.value
                        
                    selector = self.generate_selector(tag)
                    style_map[selector] = style_dict
                except:
                    pass
        
        # Extract styles from style tags
        for style_tag in soup.find_all('style'):
            try:
                sheet = cssutils.parseString(style_tag.string)
                for rule in sheet:
                    if rule.type == rule.STYLE_RULE:
                        style_dict = {}
                        for prop in rule.style:
                            style_dict[prop.name] = prop.value

                        style_map[rule.selectorText] = style_dict
            except:
                pass
                
        return style_map
    
    def generate_selector(self, tag):
        """Generate CSS selector for a tag"""
        if tag.has_attr('id'):
            return f"#{tag['id']}"
        
        path = []
        current = tag
        while current and current.name != '[document]':
            if current.has_attr('class'):
                class_str = '.'.join(current['class'])
                selector = f"{current.name}.{class_str}"
            else:
                selector = current.name

            siblings = [sib for sib in current.parent.children if isinstance(sib, Tag) and sib.name == current.name]
            if len(siblings) > 1:
                index = siblings.index(current) + 1
                selector = f"{selector}:nth-of-type({index})"
                
            path.insert(0, selector)
            current = current.parent
            
        return ' > '.join(path)
    
    def get_element_style(self, element, style_map):
        """Get computed style for an element"""
        style = {}
        
        # 1. Inline styles
        if element.has_attr('style'):
            try:
                inline_style = cssutils.parseStyle(element['style'])
                for prop in inline_style:
                    style[prop.name] = prop.value
            except:
                pass
        
        # 2. Styles from style map
        selector = self.generate_selector(element)
        if selector in style_map:
            for prop, value in style_map[selector].items():
                if prop not in style:
                    style[prop] = value
        
        return style
    
    def compare_styles(self, style1, style2, element_type):
        """Compare styles between two elements"""
        if not style1 and not style2:
            return 1.0
            
        weights = self.style_weights.get(element_type, {})
        if not weights:
            return self.similar(json.dumps(style1), json.dumps(style2))
            
        score = 0
        total_weight = 0
        
        for prop, weight in weights.items():
            total_weight += weight
            if prop in style1 and prop in style2:
                if style1[prop] == style2[prop]:
                    score += weight
                else:
                    # For numeric properties, calculate similarity
                    if any(unit in style1[prop] for unit in ['px', 'em', 'rem', '%']) and \
                       any(unit in style2[prop] for unit in ['px', 'em', 'rem', '%']):
                        
                        # Extract numeric part
                        val1 = float(re.search(r'[\d\.]+', style1[prop]).group())
                        val2 = float(re.search(r'[\d\.]+', style2[prop]).group())
                        
                        # Calculate similarity based on numeric proximity
                        if max(val1, val2) > 0:
                            similarity = min(val1, val2) / max(val1, val2)
                            score += weight * similarity
                    else:
                        # Text property similarity
                        score += weight * self.similar(style1[prop], style2[prop])
            elif prop not in style1 and prop in style2:
                continue  # Reference HTML has property but generated doesn't
            elif prop not in style1 and prop not in style2:
                total_weight -= weight  # Neither has the property
        
        return score / total_weight if total_weight > 0 else 0
    
    def extract_image_url_info(self, url):
        """Extract key information from image URL for comparison"""
        if not url:
            return None
            
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        
        # Check for specific format
        category_match = re.search(r'folder1/([^\.]+)\.jpg', path)
        if category_match:
            return category_match.group(1)
            
        # Return filename as secondary comparison
        return os.path.basename(path)
    
    def compare_images(self, img1, img2, style_map1, style_map2):
        """Compare similarity between two image elements"""
        # 1. Check image URL
        src1 = img1.get('src', '')
        src2 = img2.get('src', '')
        
        # Extract URL features
        img_info1 = self.extract_image_url_info(src1)
        img_info2 = self.extract_image_url_info(src2)
        
        # URL similarity - strict comparison for specific URL formats
        url_similarity = 1.0 if img_info1 == img_info2 else 0.0
        
        # 2. Compare styles
        style1 = self.get_element_style(img1, style_map1)
        style2 = self.get_element_style(img2, style_map2)
        
        style_similarity = self.compare_styles(style1, style2, 'image')
        
        # 3. Compare alt text
        alt1 = img1.get('alt', '')
        alt2 = img2.get('alt', '')
        alt_similarity = self.similar(alt1, alt2)
        
        # Combined score
        return 0.6 * url_similarity + 0.3 * style_similarity + 0.1 * alt_similarity
    
    def compare_text_elements(self, elem1, elem2, style_map1, style_map2):
        """Compare similarity between two text elements"""
        # 1. Content similarity
        text1 = elem1.get_text(strip=True)
        text2 = elem2.get_text(strip=True)
        content_similarity = self.similar(text1, text2)
        
        # 2. Style similarity
        style1 = self.get_element_style(elem1, style_map1)
        style2 = self.get_element_style(elem2, style_map2)
        style_similarity = self.compare_styles(style1, style2, 'text')
        
        return content_similarity, style_similarity
    
    def compare_form_elements(self, elem1, elem2, style_map1, style_map2):
        """Compare similarity between two form elements"""
        # Attributes to compare for different form elements
        attr_to_compare = {
            'form': ['action', 'method', 'enctype', 'accept-charset', 'autocomplete', 'name', 'target', 'novalidate'],
            'input': ['type', 'name', 'value', 'placeholder', 'required', 'readonly', 'disabled', 'autocomplete', 'list', 'min', 'max', 'step', 'multiple', 'pattern', 'size', 'maxlength', 'accept'],
            'button': ['type', 'name', 'value', 'disabled', 'form', 'formaction', 'formenctype', 'formmethod', 'formnovalidate', 'formtarget'],
            'select': ['name', 'multiple', 'size', 'required', 'disabled', 'autocomplete'],
            'textarea': ['name', 'rows', 'cols', 'placeholder', 'required', 'readonly', 'disabled', 'autocomplete', 'maxlength'],
            'option': ['value', 'selected', 'disabled', 'label'],
            'label': ['for'],
            'fieldset': ['name', 'disabled', 'form'],
            'legend': [],
            'datalist': ['id'],
            'output': ['for', 'name', 'form'],
            'checkbox': ['name', 'value', 'checked', 'required', 'disabled', 'form'],
            'radio': ['name', 'value', 'checked', 'required', 'disabled', 'form'],
            'file': ['name', 'accept', 'multiple', 'required', 'disabled', 'form'],
            'submit': ['name', 'value', 'disabled', 'form', 'formaction', 'formenctype', 'formmethod', 'formnovalidate', 'formtarget'],
            'reset': ['name', 'value', 'disabled', 'form'],
            'optgroup': ['label', 'disabled'],
            'meter': ['value', 'min', 'max', 'low', 'high', 'optimum', 'form'],
            'progress': ['value', 'max', 'form'],
            'keygen': ['name', 'challenge', 'keytype', 'autofocus', 'form', 'disabled'],
            'details': ['open'],
            'summary': [],
            'search': ['name', 'placeholder', 'autocomplete', 'disabled', 'form', 'maxlength', 'minlength', 'pattern', 'required', 'size'],
            'iframe': ['src', 'srcdoc', 'name', 'sandbox', 'width', 'height', 'allow', 'allowfullscreen', 'loading', 'referrerpolicy'],
            'switch': ['checked', 'disabled', 'name', 'value', 'required', 'form'],
            'range': ['name', 'min', 'max', 'step', 'value', 'disabled', 'form', 'list'],
            'colorpicker': ['name', 'value', 'disabled', 'form', 'autocomplete']
        }
        
        tag_name = elem2.name # Use reference HTML element's tag name
        
        # 1. Attribute similarity
        attrs = attr_to_compare.get(tag_name, [])
        attr_scores = []
        
        for attr in attrs:
            val1 = elem1.get(attr)
            val2 = elem2.get(attr)
            
            if val1 is not None and val2 is not None:
                attr_scores.append(self.similar(val1, val2))
            elif val1 is None and val2 is not None:
                attr_scores.append(0.0)  # Reference HTML has attribute but generated doesn't
        
        attr_similarity = sum(attr_scores) / len(attr_scores) if attr_scores else 1.0
        
        # 2. Style similarity
        style1 = self.get_element_style(elem1, style_map1)
        style2 = self.get_element_style(elem2, style_map2)
        style_similarity = self.compare_styles(style1, style2, 'form')
        
        # 3. Text content similarity (for buttons, labels, etc.)
        if tag_name in ['button', 'label', 'option']:
            text1 = elem1.get_text(strip=True)
            text2 = elem2.get_text(strip=True)
            text_similarity = self.similar(text1, text2)
            return 0.4 * attr_similarity + 0.3 * style_similarity + 0.3 * text_similarity
        
        return 0.6 * attr_similarity + 0.4 * style_similarity
    
    def match_elements(self, elements1, elements2, compare_func, style_map1, style_map2, threshold=0.9):
        """Find best matches between two element lists"""
        matched_pairs = []
        unmatched_count_2 = 0
        matched_indices2 = set()
        
        for elem1 in elements1:
            best_match = None
            best_score = threshold  # Set threshold
            best_idx = -1
            
            for i, elem2 in enumerate(elements2):
                if i in matched_indices2:
                    continue  # Skip already matched elements
                
                if elem1.name != elem2.name:
                    continue  # Only compare elements of same type
                
                # Use different comparison functions for different element types
                if compare_func == "text":
                    content_sim, style_sim = self.compare_text_elements(elem1, elem2, style_map1, style_map2)
                    score = content_sim  # For text elements, content similarity is primary
                    
                    if score > best_score:
                        best_score = score
                        best_match = (elem2, content_sim, style_sim)
                        best_idx = i
                        
                elif compare_func == "image":
                    score = self.compare_images(elem1, elem2, style_map1, style_map2)
                    
                    if score > best_score:
                        best_score = score
                        best_match = (elem2, score)
                        best_idx = i
                        
                elif compare_func == "form":
                    score = self.compare_form_elements(elem1, elem2, style_map1, style_map2)
                    
                    if score > best_score:
                        best_score = score
                        best_match = (elem2, score)
                        best_idx = i
            
            if best_match:
                matched_pairs.append((elem1, best_match))
                matched_indices2.add(best_idx)
        
        # Calculate number of unmatched elements in reference HTML
        unmatched_count_2 = len(elements2) - len(matched_indices2)
        
        return matched_pairs, unmatched_count_2
    
    def compare_html_files(self, html1, html2):
        """Compare similarity between two HTML files"""
        # html1 is generated HTML, html2 is reference HTML
        # Read HTML content
        content1 = html1
        content2 = html2
        
        # If inputs are file paths rather than HTML content
        if os.path.exists(html1):
            with open(html1, 'r', encoding='utf-8') as file:
                content1 = file.read()
        
        if os.path.exists(html2):
            with open(html2, 'r', encoding='utf-8') as file:
                content2 = file.read()
        
        # Parse HTML into trees
        soup1 = BeautifulSoup(content1, 'html.parser')
        soup2 = BeautifulSoup(content2, 'html.parser')
        
        # Extract style information
        style_map1 = self.extract_computed_styles(content1)
        style_map2 = self.extract_computed_styles(content2)
        
        # 1. Structure similarity analysis using LCS
        structure1 = self.extract_structure(soup1)
        structure2 = self.extract_structure(soup2)
        
        sequence1 = self.structure_to_sequence(structure1)
        sequence2 = self.structure_to_sequence(structure2)
        
        structure_similarity = self.calculate_structure_similarity(sequence1, sequence2)
        
        # 2. Text element analysis
        text_elements1 = soup1.find_all(self.text_elements)
        text_elements2 = soup2.find_all(self.text_elements)
        
        text_matches, unmatched_text_count = self.match_elements(
            text_elements1, text_elements2, "text", style_map1, style_map2, threshold=0.9
        )
        
        # Calculate text implementation rate
        total_text_elements = len(text_elements2)
        matched_text_elements = len(text_matches)
        text_implementation_rate = matched_text_elements / total_text_elements if total_text_elements > 0 else 1.0
        
        # Extract matched text element similarity scores
        text_content_scores = [match[1][1] for match in text_matches]  # content_sim
        text_style_scores = [match[1][2] for match in text_matches]    # style_sim
        
        text_content_similarity = sum(text_content_scores) / len(text_content_scores) if text_content_scores else 0
        text_style_similarity = sum(text_style_scores) / len(text_style_scores) if text_style_scores else 0
        
        # 3. Image element analysis
        img_elements1 = soup1.find_all('img')
        img_elements2 = soup2.find_all('img')
        
        img_matches, unmatched_img_count = self.match_elements(
            img_elements1, img_elements2, "image", style_map1, style_map2, threshold=0.9
        )
        
        # Calculate image implementation rate
        total_img_elements = len(img_elements2)
        matched_img_elements = len(img_matches)
        image_implementation_rate = matched_img_elements / total_img_elements if total_img_elements > 0 else 1.0
        
        # Extract matched image similarity scores
        img_scores = [match[1][1] for match in img_matches]
        image_similarity = sum(img_scores) / len(img_scores) if img_scores else 0
        
        # 4. Form element analysis
        form_elements1 = soup1.find_all(self.form_elements)
        form_elements2 = soup2.find_all(self.form_elements)
        
        form_matches, unmatched_form_count = self.match_elements(
            form_elements1, form_elements2, "form", style_map1, style_map2, threshold=0.9
        )
        
        # Calculate form implementation rate
        total_form_elements = len(form_elements2)
        matched_form_elements = len(form_matches)
        form_implementation_rate = matched_form_elements / total_form_elements if total_form_elements > 0 else 1.0
        
        # Extract matched form similarity scores
        form_scores = [match[1][1] for match in form_matches]
        form_similarity = sum(form_scores) / len(form_scores) if form_scores else 0
        
        # 5. Apply implementation rates to similarity scores
        text_content_similarity *= text_implementation_rate
        text_style_similarity *= text_implementation_rate
        image_similarity *= image_implementation_rate
        form_similarity *= form_implementation_rate
        
        # 6. Calculate overall similarity
        results = {
            'structure_similarity': structure_similarity,
            'text_content_similarity': text_content_similarity,
            'text_style_similarity': text_style_similarity,
            'image_similarity': image_similarity,
            'form_similarity': form_similarity,
            'text_implementation_rate': text_implementation_rate,
            'image_implementation_rate': image_implementation_rate,
            'form_implementation_rate': form_implementation_rate
        }
        
        # Calculate weighted total score
        total_similarity = sum(score * self.result_weights[key] for key, score in results.items() if key != 'details')
        results['total_similarity'] = total_similarity
        
        return results

# Add a new HTML comparison function specifically for content rather than file paths
def compare_html_files_content(self, html_content1, html_content2):
    """Compare similarity between two HTML content strings directly"""
    # Use provided HTML content directly
    content1 = html_content1
    content2 = html_content2
    
    # Parse HTML
    soup1 = BeautifulSoup(content1, 'html.parser')
    soup2 = BeautifulSoup(content2, 'html.parser')
    
    # The rest of the code is identical to compare_html_files
    # Extract style information
    style_map1 = self.extract_computed_styles(content1)
    style_map2 = self.extract_computed_styles(content2)
    
    # 1. Structure similarity analysis
    structure1 = self.extract_structure(soup1)
    structure2 = self.extract_structure(soup2)
    
    sequence1 = self.structure_to_sequence(structure1)
    sequence2 = self.structure_to_sequence(structure2)
    
    structure_similarity = self.calculate_structure_similarity(sequence1, sequence2)
    
    # 2. Text element analysis
    text_elements1 = soup1.find_all(self.text_elements)
    text_elements2 = soup2.find_all(self.text_elements)
    
    text_matches, unmatched_text_count = self.match_elements(
        text_elements1, text_elements2, "text", style_map1, style_map2, threshold=0.9
    )
    
    # Calculate text implementation rate
    total_text_elements = len(text_elements2)
    matched_text_elements = len(text_matches)
    text_implementation_rate = matched_text_elements / total_text_elements if total_text_elements > 0 else 1.0
    
    # Extract matched text element similarity scores
    text_content_scores = [match[1][1] for match in text_matches]  # content_sim
    text_style_scores = [match[1][2] for match in text_matches]    # style_sim
    
    text_content_similarity = sum(text_content_scores) / len(text_content_scores) if text_content_scores else 0
    text_style_similarity = sum(text_style_scores) / len(text_style_scores) if text_style_scores else 0
    
    # 3. Image element analysis
    img_elements1 = soup1.find_all('img')
    img_elements2 = soup2.find_all('img')
    
    img_matches, unmatched_img_count = self.match_elements(
        img_elements1, img_elements2, "image", style_map1, style_map2, threshold=0.9
    )
    
    # Calculate image implementation rate
    total_img_elements = len(img_elements2)
    matched_img_elements = len(img_matches)
    image_implementation_rate = matched_img_elements / total_img_elements if total_img_elements > 0 else 1.0
    
    # Extract matched image similarity scores
    img_scores = [match[1][1] for match in img_matches]
    image_similarity = sum(img_scores) / len(img_scores) if img_scores else 0
    
    # 4. Form element analysis
    form_elements1 = soup1.find_all(self.form_elements)
    form_elements2 = soup2.find_all(self.form_elements)
    
    form_matches, unmatched_form_count = self.match_elements(
        form_elements1, form_elements2, "form", style_map1, style_map2, threshold=0.9
    )
    
    # Calculate form implementation rate
    total_form_elements = len(form_elements2)
    matched_form_elements = len(form_matches)
    form_implementation_rate = matched_form_elements / total_form_elements if total_form_elements > 0 else 1.0
    
    # Extract matched form similarity scores
    form_scores = [match[1][1] for match in form_matches]
    form_similarity = sum(form_scores) / len(form_scores) if form_scores else 0
    
    # 5. Apply implementation rates to similarity scores
    text_content_similarity *= text_implementation_rate
    text_style_similarity *= text_implementation_rate
    image_similarity *= image_implementation_rate
    form_similarity *= form_implementation_rate
    
    # 6. Calculate overall similarity
    results = {
        'structure_similarity': structure_similarity,
        'text_content_similarity': text_content_similarity,
        'text_style_similarity': text_style_similarity,
        'image_similarity': image_similarity,
        'form_similarity': form_similarity,
        'text_implementation_rate': text_implementation_rate,
        'image_implementation_rate': image_implementation_rate,
        'form_implementation_rate': form_implementation_rate
    }
    
    # Calculate weighted total score
    total_similarity = sum(score * self.result_weights[key] for key, score in results.items() if key != 'details')
    results['total_similarity'] = total_similarity
    
    return results

# Add the new function to HTMLComparer class
HTMLComparer.compare_html_files_content = compare_html_files_content

# Main function
def main():
    # Models and datasets to test
    model_names = ["o1", "o4-mini"]
    dataset_names = ["Image_to_code", "Interaction_Authoring", "Text_to_code"]
    
    # Set input and output directories with relative paths
    input_dir = "./data"  # Directory containing JSON files
    output_dir = "./results"  # Output directory
    
    # Process JSON files
    process_json_files(input_dir, output_dir, model_names, dataset_names)

if __name__ == "__main__":
    main() 