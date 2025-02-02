import requests
from tqdm import tqdm

# File to download
#we will need to download more files accordingly 
path = "crawl-data/CC-MAIN-2024-51/segments/1733066362401.69/wet/CC-MAIN-20241205180803-20241205210803-00720.warc.wet.gz"
base_url = "https://data.commoncrawl.org/"
file_url = base_url + path

# Output file path
output_file = "test_file.wet.gz"

try:
    print(f"Downloading: {file_url}")
    
    # Send a HEAD request to get the file size
    response = requests.head(file_url)
    file_size = int(response.headers.get('content-length', 0))

    # Download the file with a progress bar
    with requests.get(file_url, stream=True) as response, open(output_file, "wb") as f, tqdm(
        total=file_size, unit='B', unit_scale=True, desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            progress_bar.update(len(chunk))
    
    print(f"Downloaded successfully: {output_file}")
except requests.exceptions.RequestException as e:
    print(f"Error during download: {e}")
