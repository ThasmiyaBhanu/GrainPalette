from utils.data_splitter import RiceDataSplitter
import os

def main():
    # Define paths
    directory = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(directory, "..", "Rice_Image_Dataset", "Rice_Image_Dataset")
    destination_path = os.path.join(directory, "..", "Data")
    
    # Create the data splitter
    splitter = RiceDataSplitter(source_path, destination_path)
    
    # Split the data (80% train, 20% test)
    print("Starting data splitting process...")
    train_path, test_path = splitter.split_data(test_size=0.2, random_state=42)
    
    # Verify the split
    splitter.verify_split()
    
    print(f"\nData splitting completed!")
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")

if __name__ == "__main__":
    main()

