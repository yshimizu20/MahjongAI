import os
import re
import requests

dir_name = "raw/2021/"
output_dir_name = "processed/2021/"

# Collect and process URLs in each HTML file
for filename in os.listdir(dir_name):
    if filename.endswith(".html"):
        filepath = os.path.join(dir_name, filename)
        with open(filepath, "r") as file:
            html_content = file.read()

        # Find all URLs in <a> tags using regular expression
        urls = re.findall(r'<a\s+[^>]*?href=[\'"](.*?)[\'"].*?>', html_content)

        # Process and download data from each URL
        for url in urls:
            if url.startswith("http://tenhou.net/0/?log="):
                # Modify the URL
                gid = url.split("=")[1]
                modified_url = "https://tenhou.net/0/log/?" + gid
                # print(f"Downloading data from: {modified_url}")

                # Send a GET request to the modified URL
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                }
                response = requests.get(modified_url, headers=headers)
                if response.status_code == 200:
                    # Save the response content to a file
                    data_filename = os.path.join(output_dir_name, gid + ".log")
                    with open(data_filename, "wb") as data_file:
                        data_file.write(response.content)
                else:
                    print(f"Failed to download data from: {modified_url}")
