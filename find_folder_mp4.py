import os
import shutil


def copy_media_files(src_dir, dst_dir):
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        raise Exception(f"Source directory {src_dir} does not exist.")

    # Create the destination directory if it does not exist
    os.makedirs(dst_dir, exist_ok=True)

    # Define the file extensions to look for
    file_extensions = [".mp4"]

    # Iterate over the files in the source directory
    for root, _, files in os.walk(src_dir):
        for file in files:
            # Check if the file has one of the desired extensions
            if any(file.endswith(ext) for ext in file_extensions):
                # Construct full file path
                src_file = os.path.join(root, file)
                # Copy the file to the destination directory
                shutil.copy(src_file, dst_dir)
                print(f"Copied {file} to {dst_dir}")

    print(f"All media files have been copied to {dst_dir}")


# Example usage
src_directory = "experience"
dst_directory = "mp4_folder_experience"

copy_media_files(src_directory, dst_directory)
