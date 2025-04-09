import os
import gdown

# URL of the Google Drive folder you provided
folder_url = "https://drive.google.com/drive/folders/1mh3u7JcFji9V1JJJowZhObiihjAn77VA?usp=sharing"

# Define the output folder where the files will be saved
output_folder = "mdls"

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")
else:
    print(f"Folder exists: {output_folder}")

# Use gdown to download the entire folder
gdown.download_folder(url=folder_url, output=output_folder, quiet=False)