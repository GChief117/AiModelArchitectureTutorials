import json
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup

# Path to your WARC file
warc_file_path = r"C:\Users\gunne\OneDrive\Desktop\testing\test_file_warc.warc"

extracted_records = []

with open(warc_file_path, 'rb') as stream:
    for record in ArchiveIterator(stream):
        # Process only 'response' records (which contain web content)
        if record.rec_type == 'response':
            # Extract the URL from the WARC record headers
            url = record.rec_headers.get_header('WARC-Target-URI')

            # Read the HTTP response payload
            payload_bytes = record.content_stream().read()

            # Try decoding the payload (adjust encoding as necessary)
            try:
                payload_str = payload_bytes.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                payload_str = payload_bytes.decode('latin-1', errors='replace')

            # Use BeautifulSoup to extract text from HTML
            soup = BeautifulSoup(payload_str, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

            # Create a dictionary for this record
            record_data = {
                'url': url,
                'text': text
            }
            extracted_records.append(record_data)

# Convert the list of records to JSON format
json_output = json.dumps(extracted_records, indent=2, ensure_ascii=False)

# Write JSON output to a file
output_file = "output.json"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(json_output)

print(f"Extraction complete! JSON output written to {output_file}")
