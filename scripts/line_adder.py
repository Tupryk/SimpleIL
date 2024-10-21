import os

def process_txt_file(file_path):
    """Reads the last line of a .txt file, processes it, and appends the new line."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        if not lines:
            print(f"No data in {file_path}. Skipping.")
            return

        # Get the last line and split into numbers
        last_line = lines[-1].strip()
        numbers = last_line.split()

        if len(numbers) != 8:
            print(f"Incorrect format in {file_path}. Expected 8 numbers, found {len(numbers)}. Skipping.")
            return

        # Modify the line: replace the 8th number with 0
        new_line = numbers[:7] + ['0']

        # Create a string of the new line
        new_line_str = ' '.join(new_line)

        # Append the new line to the file
        with open(file_path, 'a') as file:
            file.write(new_line_str)

        print(f"Processed and updated {file_path}")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

def traverse_folders(parent_folder):
    """Traverses through all subfolders and processes .txt files."""
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file == "proprioceptive.txt":
                file_path = os.path.join(root, file)
                process_txt_file(file_path)

if __name__ == "__main__":
    # Replace this with the folder you want to start from
    parent_folder = 'rec'
    traverse_folders(parent_folder)
