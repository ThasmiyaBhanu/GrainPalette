import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class RiceDataPreprocessor:
    def __init__(self, dataset_path, img_height=224, img_width=224, batch_size=32):
        """
        Initialize the Rice Data Preprocessor
        
        Args:
            dataset_path (str): Path to the Rice_Image_Dataset directory
            img_height (int): Target height for images
            img_width (int): Target width for images
            batch_size (int): Batch size for data loading
        """
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_path = None
        self.test_path = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self.class_names = None
        
    def set_data_paths(self):
        """
        Set up train and test data paths
        """
        self.train_path = os.path.join(self.dataset_path, 'train')
        self.test_path = os.path.join(self.dataset_path, 'test')
        
        print(f"Train data path: {self.train_path}")
        print(f"Test data path: {self.test_path}")
        
        # Verify paths exist
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Train path does not exist: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test path does not exist: {self.test_path}")
            
        # Get class names from train directory
        self.class_names = os.listdir(self.train_path)
        print(f"Classes found: {self.class_names}")
        print(f"Number of classes: {len(self.class_names)}")
        
    def create_data_generators(self, validation_split=0.2):
        """
        Create ImageDataGenerator objects for training, validation, and testing
        
        Args:
            validation_split (float): Fraction of training data to use for validation
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Test data generator (only rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Create validation generator
        self.validation_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        # Create test generator
        self.test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        
    def visualize_sample_images(self, num_images=16):
        """
        Visualize sample images from the training set
        
        Args:
            num_images (int): Number of images to display
        """
        # Get a batch of images and labels
        images, labels = next(self.train_generator)
        
        # Calculate grid size
        grid_size = int(np.sqrt(num_images))
        
        plt.figure(figsize=(12, 12))
        for i in range(min(num_images, len(images))):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i])
            
            # Get class name from label
            class_idx = np.argmax(labels[i])
            class_name = list(self.train_generator.class_indices.keys())[class_idx]
            plt.title(f'Class: {class_name}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def get_class_distribution(self):
        """
        Get and visualize class distribution in the dataset
        """
        train_counts = []
        test_counts = []
        
        for class_name in self.class_names:
            train_class_path = os.path.join(self.train_path, class_name)
            test_class_path = os.path.join(self.test_path, class_name)
            
            train_count = len(os.listdir(train_class_path)) if os.path.exists(train_class_path) else 0
            test_count = len(os.listdir(test_class_path)) if os.path.exists(test_class_path) else 0
            
            train_counts.append(train_count)
            test_counts.append(test_count)
        
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'Class': self.class_names,
            'Train': train_counts,
            'Test': test_counts
        })
        
        print("Class Distribution:")
        print(df)
        
        # Visualize distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(self.class_names, train_counts, color='skyblue')
        plt.title('Training Set Distribution')
        plt.xlabel('Rice Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(self.class_names, test_counts, color='lightcoral')
        plt.title('Test Set Distribution')
        plt.xlabel('Rice Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return df
        
    def prepare_data(self, validation_split=0.2):
        """
        Complete data preparation pipeline
        
        Args:
            validation_split (float): Fraction of training data to use for validation
        """
        print("Starting data preparation...")
        
        # Set up data paths
        self.set_data_paths()
        
        # Create data generators
        self.create_data_generators(validation_split)
        
        print("Data preparation completed successfully!")
        
        return self.train_generator, self.validation_generator, self.test_generator

