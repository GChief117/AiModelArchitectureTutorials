from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
import json
import re

# File path
warc_file_path = r"C:\Users\gunne\OneDrive\Desktop\testing\test_file_warc.warc"

# Function to clean HTML and extract visible text
def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.extract()

    # Get visible text
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to extract content from WARC
def extract_warc_content(file_path):
    extracted_data = []
    with open(file_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            try:
                if record.rec_type == "response":
                    url = record.rec_headers.get("WARC-Target-URI", "Unknown URL")
                    payload = record.content_stream().read().decode("utf-8", errors="ignore")

                    # Clean HTML content
                    clean_content = clean_html(payload)

                    if clean_content.strip():  # Skip empty records
                        extracted_data.append({"url": url, "content": clean_content})
            except Exception as e:
                print(f"Skipping malformed record: {e}")
    return extracted_data

# Extract content
data = extract_warc_content(warc_file_path)

# Save to JSON
output_path = r"C:\Users\gunne\OneDrive\Desktop\testing\cleaned_warc.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Cleaned data saved to: {output_path}")
