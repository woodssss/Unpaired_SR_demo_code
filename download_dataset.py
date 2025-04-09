import os
import gdown

# URL of the Google Drive folder
folder_url = "https://drive.google.com/drive/folders/156pCMxUZJ55NvRjBGukfO9kIaV4yFrUh?usp=sharing"

# Define the output folder (named "data")
output_folder = "data"

# Create the "data" folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")
else:
    print(f"Folder exists: {output_folder}")

# Download the folder into the "data" folder
gdown.download_folder(url=folder_url, output=output_folder, quiet=False)
