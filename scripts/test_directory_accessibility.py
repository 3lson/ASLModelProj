import os

def test_directory_accessibility(directory):
    # Check if the directory exists, and create it if not
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        except Exception as e:
            print(f"Failed to create directory '{directory}': {e}")
            return False
    else:
        print(f"Directory '{directory}' already exists.")
    
    # Check if we can write a test file to the directory
    test_file_path = os.path.join(directory, 'test_file.txt')
    try:
        with open(test_file_path, 'w') as test_file:
            test_file.write("This is a test.")
        print(f"Test file written to '{test_file_path}'.")
        
        # Check if the file was created
        if os.path.exists(test_file_path):
            print("Test file successfully created.")
            # Optionally read back the file to ensure it was written correctly
            with open(test_file_path, 'r') as test_file:
                content = test_file.read()
                print(f"Content of test file: {content}")
            os.remove(test_file_path)  # Clean up after the test
            print("Test file removed.")
        else:
            print("Test file was not created.")
            
        return True
    except Exception as e:
        print(f"Failed to write to directory '{directory}': {e}")
        return False

# Run the directory accessibility test
test_directory = '/Users/elsontho/Desktop/ASLModel/models'
test_directory_accessibility(test_directory)
