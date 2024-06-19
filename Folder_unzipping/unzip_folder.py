import zipfile 
import os 

def unzip_folder(zip_path, extract_to):
    """
    Unzips a zip file to the specified directory.

    Parameters:
    - zip_path: Path to the zip file.
    - extract_to: Directory where the contents will be extracted.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f'Extracted {zip_path} to {extract_to}')
    except zipfile.BadZipFile:
        print(f'Error: {zip_path} is not a valid zip file.')
    except FileNotFoundError:
        print(f'Error: {zip_path} not found.')
    except Exception as e:
        print(f'An error occurred: {e}')

# Example usage
zip_path = 'DUTS-TR.zip'
extract_to = 'DUTS-Dataset'

os.makedirs(extract_to, exist_ok=True)
unzip_folder(zip_path, extract_to)