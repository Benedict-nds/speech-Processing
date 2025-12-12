import os

def print_directory_contents(path):
    """
    This function takes the path to a directory as input and prints the names of all files and subdirectories
    in that directory. It also handles potential errors such as the directory not existing or not being accessible.
    """
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                print(f"File: {item}")
            elif os.path.isdir(item_path):
                print(f"Directory: {item}")
            else:
                print(f"Other: {item}")
    except FileNotFoundError:
        print(f"Error: Directory not found at path: {path}")
    except PermissionError:
        print(f"Error: Permission denied to access directory: {path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Get the current working directory (project folder)
project_folder = os.getcwd()

# Print the contents of the project folder
print_directory_contents(project_folder)