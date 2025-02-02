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

def process_wet_file(wet_file_path):
    """
    Processes a WET file and extracts data from each 'conversion' record.
    Cleans the text and detects its language (adding a 'language' field).
    
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
    print(f"Extracted and cleaned {len(data)} records from the WET file.")
    
    write_json(data, json_output_path)
    print(f"JSON output written to: {json_output_path}")
    
    write_csv(data, csv_output_path)
    print(f"CSV output written to: {csv_output_path}")
