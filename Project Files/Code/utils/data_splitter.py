import os
import shutil
from sklearn.model_selection import train_test_split
import random
import time
from pathlib import Path

class RiceDataSplitter:
    def __init__(self, source_path, destination_path):
        """
        Initialize the Rice Data Splitter
        
        Args:
            source_path (str): Path to the original Rice_Image_Dataset directory
            destination_path (str): Path where train/test split will be created
        """
        self.source_path = source_path
        self.destination_path = destination_path
        self.train_path = os.path.join(destination_path, 'train')
        self.test_path = os.path.join(destination_path, 'test')
        
    def create_directories(self):
        """
        Create train and test directories structure
        """
        # Create main directories
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        
        # Get class names from source directory
        class_names = [d for d in os.listdir(self.source_path) 
                      if os.path.isdir(os.path.join(self.source_path, d))]
        
        # Create class subdirectories in train and test
        for class_name in class_names:
            os.makedirs(os.path.join(self.train_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(self.test_path, class_name), exist_ok=True)
            
        print(f"Created directories for classes: {class_names}")
        return class_names
    
    def _safe_copy(self, src, dst, max_retries=3):
        """
        Safely copy a file with retry logic
        
        Args:
            src (str): Source file path
            dst (str): Destination file path
            max_retries (int): Maximum number of retry attempts
        """
        for attempt in range(max_retries):
            try:
                shutil.copy2(src, dst)
                return True
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"Permission error copying {os.path.basename(src)}, retrying in 1 second... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                else:
                    print(f"Failed to copy {os.path.basename(src)} after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                print(f"Unexpected error copying {os.path.basename(src)}: {e}")
                return False
        return False
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into train and test sets
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        print(f"Starting data split with test_size={test_size}...")
        
        # Create directory structure
        class_names = self.create_directories()
        
        total_images = 0
        train_images = 0
        test_images = 0
        
        for class_name in class_names:
            class_source_path = os.path.join(self.source_path, class_name)
            
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(class_source_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            print(f"Processing {class_name}: {len(image_files)} images")
            
            # Split files into train and test
            train_files, test_files = train_test_split(
                image_files, 
                test_size=test_size, 
                random_state=random_state,
                stratify=None
            )
            
            # Copy training files
            train_class_path = os.path.join(self.train_path, class_name)
            train_success = 0
            for file in train_files:
                src = os.path.join(class_source_path, file)
                dst = os.path.join(train_class_path, file)
                if self._safe_copy(src, dst):
                    train_success += 1
            
            # Copy test files
            test_class_path = os.path.join(self.test_path, class_name)
            test_success = 0
            for file in test_files:
                src = os.path.join(class_source_path, file)
                dst = os.path.join(test_class_path, file)
                if self._safe_copy(src, dst):
                    test_success += 1
            
            total_images += len(image_files)
            train_images += len(train_files)
            test_images += len(test_files)
            
            print(f"  - Train: {len(train_files)} images")
            print(f"  - Test: {len(test_files)} images")
        
        print(f"\nData split completed!")
        print(f"Total images: {total_images}")
        print(f"Training images: {train_images}")
        print(f"Test images: {test_images}")
        print(f"Train/Test ratio: {train_images/total_images:.2%}/{test_images/total_images:.2%}")
        
        return self.train_path, self.test_path
    
    def verify_split(self):
        """
        Verify the data split by counting files in each directory
        """
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            print("Split directories do not exist. Please run split_data() first.")
            return
        
        print("\nData Split Verification:")
        print("-" * 40)
        
        train_classes = os.listdir(self.train_path)
        
        for class_name in train_classes:
            train_class_path = os.path.join(self.train_path, class_name)
            test_class_path = os.path.join(self.test_path, class_name)
            
            train_count = len(os.listdir(train_class_path))
            test_count = len(os.listdir(test_class_path))
            total_count = train_count + test_count
            
            print(f"{class_name}:")
            print(f"  - Train: {train_count} ({train_count/total_count:.1%})")
            print(f"  - Test: {test_count} ({test_count/total_count:.1%})")
            print(f"  - Total: {total_count}")
            print()

