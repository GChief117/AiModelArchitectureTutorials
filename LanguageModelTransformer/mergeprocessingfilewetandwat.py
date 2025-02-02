import csv
import json
import re
from warcio.archiveiterator import ArchiveIterator
from langdetect import detect, DetectorFactory
from tqdm import tqdm  # For the progress bar
import hashlib

# Set a seed for reproducibility in language detection.
DetectorFactory.seed = 0

def clean_text(text: str) -> str:
    """
    Cleans the provided text by:
      - Removing any lingering HTML tags (if present)
      - Removing URLs
      - Lowercasing the text
      - Removing extra whitespace and control characters.
      
    Args:
        text (str): The original text.
        
    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'<[^>]+>', '', text)   # Remove HTML tags
    text = re.sub(r'http\S+', '', text)     # Remove URLs
    text = text.lower()                     # Lowercase
    text = re.sub(r'[\r\n\t]+', ' ', text)   # Replace line breaks/tabs with space
    text = re.sub(r'\s+', ' ', text)         # Collapse multiple spaces
    return text.strip()

def process_wet_file(wet_file_path):
    """
    Processes a WET file and extracts data from each 'conversion' record.
    Cleans the text and detects its language.
    Only English records are kept.
    
    Returns:
        dict: A dictionary mapping URL to a record with keys 'url', 'date', 'language', 'text'.
    """
    wet_data = {}

    with open(wet_file_path, 'rb') as stream:
        for record in tqdm(ArchiveIterator(stream), desc="Processing WET Records"):
            if record.rec_type != 'conversion':
                continue

            url = record.rec_headers.get_header('WARC-Target-URI')
            date = record.rec_headers.get_header('WARC-Date')
            
            text_bytes = record.content_stream().read()
            try:
                text = text_bytes.decode('utf-8', errors='replace').strip()
            except Exception:
                continue  # Skip if cannot decode
            
            text = clean_text(text)
            if len(text) < 20 or len(re.findall(r'[a-zA-Z]', text)) < 3:
                continue  # Skip very short or insufficient text
            
            try:
                language = detect(text)
            except Exception:
                continue
            
            if language != 'en':
                continue  # Keep only English records

            wet_data[url] = {
                'url': url,
                'date': date,
                'language': language,
                'text': text
            }
    return wet_data

def extract_wat_data(wat_file_path):
    """
    Processes a WAT file and extracts metadata from each 'metadata' record.
    Uses the JSON content from the WAT record.
    
    Returns:
        dict: A dictionary mapping URL to metadata. In this version, we extract the metadata
              records from Envelope->Payload-Metadata->WARC-Metadata-Metadata->Metadata-Records.
    """
    wat_data = {}
    
    with open(wat_file_path, 'rb') as stream:
        for record in tqdm(ArchiveIterator(stream), desc="Processing WAT Records"):
            if record.rec_type != 'metadata':
                continue

            url = record.rec_headers.get_header('WARC-Target-URI')
            content_bytes = record.content_stream().read()
            try:
                content_json = json.loads(content_bytes.decode('utf-8', errors='replace'))
            except Exception:
                continue
            
            # Extract metadata records from the JSON structure.
            metadata_records = content_json.get("Envelope", {}) \
                                           .get("Payload-Metadata", {}) \
                                           .get("WARC-Metadata-Metadata", {}) \
                                           .get("Metadata-Records", [])
            
            wat_data[url] = {
                'wat_metadata': metadata_records
            }
    return wat_data

def generate_id(url: str) -> str:
    """
    Generates a unique ID based on the URL.
    """
    return hashlib.sha1(url.encode('utf-8')).hexdigest()

def merge_data(wet_data, wat_data):
    """
    Merges the WET and WAT datasets based on URL.
    Each merged record contains:
      - id
      - url
      - date
      - language
      - text (from WET)
      - wat_metadata (from WAT, if available; otherwise empty list)
    
    Returns:
        list: A list of merged record dictionaries.
    """
    merged_data = []
    
    for url, wet_record in tqdm(wet_data.items(), desc="Merging Data"):
        # Use the URL as the key to find matching metadata.
        wat_record = wat_data.get(url, {})
        merged_entry = {
            "id": generate_id(url),
            "url": wet_record["url"],
            "date": wet_record["date"],
            "language": wet_record["language"],
            "text": wet_record["text"],
            "wat_metadata": wat_record.get("wat_metadata", []),
            "metadata": {
                "source": "Common Crawl"
            }
        }
        merged_data.append(merged_entry)
    
    return merged_data

def write_json(data, json_file_path):
    """Writes merged data to a JSON file."""
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(data, csv_file_path):
    """
    Writes merged data to a CSV file.
    Note: Complex fields like wat_metadata are converted to JSON strings.
    """
    fieldnames = ["id", "url", "date", "language", "text", "wat_metadata", "metadata"]
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            # Convert complex fields to JSON strings.
            entry_copy = entry.copy()
            entry_copy["wat_metadata"] = json.dumps(entry_copy.get("wat_metadata", []))
            entry_copy["metadata"] = json.dumps(entry_copy.get("metadata", {}))
            writer.writerow(entry_copy)

if __name__ == "__main__":
    # Update these paths to your local files.
    wet_file_path = r"C:\Users\gunne\OneDrive\Desktop\testing\test_wetfile.wet"
    wat_file_path = r"C:\Users\gunne\OneDrive\Desktop\testing\test_watfile.wat"
    
    json_output_path = "merged_output.json"
    csv_output_path = "merged_output.csv"
    
    print(f"Processing WET file: {wet_file_path}")
    wet_data = process_wet_file(wet_file_path)
    print(f"Extracted {len(wet_data)} records from WET file.")
    
    print(f"Processing WAT file: {wat_file_path}")
    wat_data = extract_wat_data(wat_file_path)
    print(f"Extracted metadata for {len(wat_data)} records from WAT file.")
    
    merged_data = merge_data(wet_data, wat_data)
    print(f"Merged total records: {len(merged_data)}")
    
    write_json(merged_data, json_output_path)
    print(f"JSON output written to: {json_output_path}")
    
    write_csv(merged_data, csv_output_path)
    print(f"CSV output written to: {csv_output_path}")
