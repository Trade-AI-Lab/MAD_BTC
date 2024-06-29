import os


def update_readme_with_media(src_dir, readme_file):
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        raise Exception(f"Source directory {src_dir} does not exist.")

    # Placeholder text in the Markdown file where media will be inserted
    placeholder = "# Video MP4"

    # Read the existing Markdown file or create it if it does not exist
    if os.path.exists(readme_file):
        with open(readme_file, "r") as file:
            lines = file.readlines()
    else:
        lines = [placeholder]

    # Prepare the HTML lines to insert
    media_lines = []
    for file_name in sorted(os.listdir(src_dir)):
        if file_name.endswith((".mp4", ".gif", ".jpg")):
            file_path = os.path.join(src_dir, file_name).replace("\\", "/")
            if file_name.endswith(".mp4"):
                media_lines.append(
                    f'<video width="640" height="480" controls>\n  <source src="{file_path}" type="video/mp4">\n  Your browser does not support the video tag.\n</video>\n\n'
                )
            elif file_name.endswith(".gif"):
                media_lines.append(f'<img src="{file_path}" width="640"/>\n\n')
            elif file_name.endswith(".jpg"):
                media_lines.append(f'<img src="{file_path}" width="640"/>\n\n')

    # Insert the HTML lines at the placeholder location
    with open(readme_file, "w") as file:
        for line in lines:
            file.write(line)
            if placeholder in line:
                file.writelines(media_lines)

    print(f"Markdown file {readme_file} updated successfully.")


# Example usage
src_directory = "mp4_folder_experience"
readme_file = "README.md"

update_readme_with_media(src_directory, readme_file)
