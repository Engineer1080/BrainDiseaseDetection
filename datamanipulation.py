import os
import hashlib


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def remove_duplicates(directory_path):
    file_hashes = {}
    duplicate_count = 0  # Counter for duplicate files
    for folder_name, _, file_names in os.walk(directory_path):
        for filename in file_names:
            file_path = os.path.join(folder_name, filename)  # path to each individual image file
            file_hash = calculate_md5(file_path)
            if file_hash not in file_hashes:
                file_hashes[file_hash] = filename
            else:
                print(f'Duplicate found: {filename} is a duplicate of {file_hashes[file_hash]}')
                duplicate_count += 1  # Increment the counter for each found duplicate
                os.remove(file_path)  # Remove the duplicate file
    return duplicate_count


# Replace these paths with your actual paths
train_directory_path = 'Training'
test_directory_path = 'Testing'


# Remove duplicates in training and testing directories
train_duplicates = remove_duplicates(train_directory_path)
test_duplicates = remove_duplicates(test_directory_path)

# Print the duplicate counts
print(f'Number of duplicate files removed from training directory: {train_duplicates}')
print(f'Number of duplicate files removed from testing directory: {test_duplicates}')


def count_files_in_subdirectories(directory_path):
    print(f'{directory_path} directory:')
    for folder_name in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(subdirectory_path):
            file_count = len(os.listdir(subdirectory_path))
            print(f'The folder {folder_name} contains {file_count} files.')


# Count the number of files in each subdirectory of the "Training" directory
count_files_in_subdirectories('Training')
count_files_in_subdirectories('Testing')
