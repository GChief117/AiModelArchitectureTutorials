import csv
import json
import re
from warcio.archiveiterator import ArchiveIterator
from langdetect import detect, DetectorFactory
from tqdm import tqdm  # For the progress bar

# Set a seed for reproducibility in language detection.
DetectorFactory.seed = 0

def clean_text(text: str) -> str:
    """
    Cleans the provided text by:
      - Removing any lingering HTML tags (if present)
      - Removing URLs
      - Lowercasing the text
      - Removing extra whitespace and control characters
      
    Args:
        text (str): The original text.
        
    Returns:
        str: The cleaned text.
    """
    # Remove HTML tags (if any)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Lowercase the text
    text = text.lower()
    
    # Replace line breaks, tabs, and multiple spaces with a single space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def is_mostly_english(text: str, threshold: float = 0.9) -> bool:
    """
    Determines if the text is mostly English based on its character composition.
    
    It compares the number of Latin (A-Z, a-z) alphabetic characters to the total 
    number of alphabetic characters (ignoring digits and underscores).
    
    Args:
        text (str): The text to check.
        threshold (float): The minimum ratio of Latin letters to total letters required.
        
    Returns:
        bool: True if the ratio is above the threshold, False otherwise.
    """
    # Get all letters (ignoring digits, punctuation, and underscores)
    all_letters = re.findall(r'[^\W\d_]', text, re.UNICODE)
    # Get only Latin letters (A-Z, a-z)
    latin_letters = re.findall(r'[A-Za-z]', text)
    
    if not all_letters:
        return False
    ratio = len(latin_letters) / len(all_letters)
    return ratio >= threshold

def is_probably_code(text: str) -> bool:
    """
    A simple heuristic to check if a text snippet is likely code.
    
    This function looks for common code keywords and symbols. Adjust the keywords 
    and threshold as needed.
    
    Args:
        text (str): The text to check.
        
    Returns:
        bool: True if the text appears to be code, False otherwise.
    """
    code_keywords = [
        'function', 'if(', 'if (', 'const', 'let', 'var', '=>', 'return',
        '{', '}', ';', '(', ')'
    ]
    count = 0
    for keyword in code_keywords:
        if keyword in text:
            count += 1
    # If 3 or more code-related indicators are found, assume it's code.
    return count >= 3

def process_wet_file(wet_file_path):
    """
    Processes a WET file and extracts data from each 'conversion' record.
    Cleans the text, detects its language (adding a 'language' field), and
    only keeps records that are English and do not appear to be code or mixed-language.
    
    Args:
        wet_file_path (str): Path to the WET file.
        
    Returns:
        list: A list of dictionaries with keys 'url', 'date', 'language', and 'text'.
    """
    records_data = []
    
    with open(wet_file_path, 'rb') as stream:
        # Wrap ArchiveIterator with tqdm to display a progress bar.
        for record in tqdm(ArchiveIterator(stream), desc="Processing Records"):
            if record.rec_type != 'conversion':
                continue

            # Extract URL and date from the record headers.
            url = record.rec_headers.get_header('WARC-Target-URI')
            date = record.rec_headers.get_header('WARC-Date')

            # Read and decode the text content.
            text_bytes = record.content_stream().read()
            try:
                text = text_bytes.decode('utf-8', errors='replace').strip()
            except Exception:
                continue  # Skip records that cannot be decoded

            # Clean the text.
            text = clean_text(text)
            
            # Skip very short texts or texts lacking enough alphabetic content.
            if len(text) < 20 or len(re.findall(r'[a-zA-Z]', text)) < 3:
                continue

            # Detect the language. If detection fails, skip the record.
            try:
                language = detect(text)
            except Exception:
                continue

            # Only keep records where the detected language is English.
            if language != 'en':
                continue

            # Skip texts that appear to be code.
            if is_probably_code(text):
                continue

            # Skip texts that are mixed-language (i.e. not mostly English based on character composition).
            if not is_mostly_english(text):
                continue

            records_data.append({
                'url': url,
                'date': date,
                'language': language,
                'text': text
            })
            
    return records_data

def write_json(data, json_file_path):
    """Writes the data to a JSON file."""
    with open(json_file_path, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

def write_csv(data, csv_file_path):
    """Writes the data to a CSV file."""
    fieldnames = ['url', 'date', 'language', 'text']
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for record in data:
            writer.writerow(record)

if __name__ == '__main__':
    # Path to your WET file.
    wet_file_path = r"C:\Users\gunne\OneDrive\Desktop\testing\test_wetfile.wet"
    
    # Define output file paths.
    json_output_path = "wet_output.json"
    csv_output_path = "wet_output.csv"
    
    print(f"Processing WET file: {wet_file_path}")
    data = process_wet_file(wet_file_path)
    print(f"Extracted and cleaned {len(data)} English records from the WET file.")
    
    write_json(data, json_output_path)
    print(f"JSON output written to: {json_output_path}")
    
    write_csv(data, csv_output_path)
    print(f"CSV output written to: {csv_output_path}")
