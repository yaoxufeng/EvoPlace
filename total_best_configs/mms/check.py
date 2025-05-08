import os

# Specify the path where your directories are located (current directory in this case)
base_directory = "."

# List all directories in the base directory
directories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

# Iterate over each directory to check for .py files
directories_without_py = []

for directory in directories:
    # Get the full path of the directory
    dir_path = os.path.join(base_directory, directory)
    
    # List all files in the directory
    files_in_directory = os.listdir(dir_path)
    
    # Check if any file in the directory ends with '.py'
    contains_py_file = any(file.endswith('.py') for file in files_in_directory)
    
    # If no .py file is found, add the directory to the list
    if not contains_py_file:
        directories_without_py.append(directory)

# Print directories that don't contain any .py files
if directories_without_py:
    print("Directories without .py files:")
    for dir_name in directories_without_py:
        print(dir_name)
else:
    print("All directories contain .py files.")
