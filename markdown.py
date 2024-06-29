import os

# Directory containing the PNG files
plot_dir = "Robustness_Plot"

# Ensure the directory exists
if not os.path.exists(plot_dir):
    raise Exception(f"Directory {plot_dir} does not exist.")

# Existing Markdown file to update
markdown_file = "README.md"

# Placeholder text in the Markdown file where images will be inserted
placeholder = "# Images of Clustering"

# Read the existing Markdown file
with open(markdown_file, "r") as file:
    lines = file.readlines()

# Prepare the image paths to insert
image_lines = []
i = 0
for file_name in sorted(os.listdir(plot_dir)):
    if file_name.endswith(".png"):
        plot_path = os.path.join(plot_dir, file_name)
        image_lines.append(f"## Example {i + 1}\n")
        image_lines.append(f'<img src="{plot_path}" width="1024"/>\n\n')
        i += 1
# Insert the image paths at the placeholder location
with open(markdown_file, "w") as file:
    for line in lines:
        file.write(line)
        if placeholder in line:
            file.writelines(image_lines)

print(f"Markdown file {markdown_file} updated successfully.")
