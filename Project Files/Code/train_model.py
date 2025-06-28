import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Import our custom modules
from preprocessing.data_preprocessing import RiceDataPreprocessor as DataPreprocessor
from model_builder import create_rice_classifier

class ModelTrainer:
    """
    Rice classification model trainer
    """
    
    def __init__(self, train_dir, test_dir, model_save_dir='../models'):
        """
        Initialize the model trainer
        
        Args:
            train_dir (str): Path to training data directory
            test_dir (str): Path to test data directory
            model_save_dir (str): Directory to save trained models
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_save_dir = model_save_dir
        self.preprocessor = None
        self.model = None
        self.history = None
        
        # Create model save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Get class names from training directory
        self.class_names = sorted([d for d in os.listdir(train_dir) 
                                  if os.path.isdir(os.path.join(train_dir, d))])
        self.num_classes = len(self.class_names)
        
        print(f"Found {self.num_classes} rice varieties: {self.class_names}")
    
    def setup_data_generators(self, batch_size=32, image_size=(224, 224), augment_data=True, validation_split=0.2):
        """
        Setup data generators for training, validation, and testing
        
        Args:
            batch_size (int): Batch size for training
            image_size (tuple): Target image size
            augment_data (bool): Whether to apply data augmentation
            validation_split (float): Fraction of training data to use for validation
        """
        self.preprocessor = DataPreprocessor(
            dataset_path="../Data",
            img_height=image_size[0],
            img_width=image_size[1],
            batch_size=batch_size
        )
        
        # Prepare data generators
        self.train_generator, self.validation_generator, self.test_generator = self.preprocessor.prepare_data(
            validation_split=validation_split
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Steps per epoch: {len(self.train_generator)}")
        print(f"Validation steps: {len(self.validation_generator)}")
    
    def create_model(self, freeze_base=True, dropout_rate=0.2, learning_rate=0.001):
        """
        Create and compile the rice classification model
        
        Args:
            freeze_base (bool): Whether to freeze base MobileNet layers
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        self.model = create_rice_classifier(
            num_classes=self.num_classes,
            input_shape=(224, 224, 3),
            freeze_base=freeze_base,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        print(f"Model created with {self.num_classes} output classes")
    
    def train_model(self, epochs=20, patience=10, save_best=True):
        """
        Train the rice classification model
        
        Args:
            epochs (int): Number of training epochs
            patience (int): Early stopping patience
            save_best (bool): Whether to save the best model
            
        Returns:
            tf.keras.History: Training history
        """
        if self.model is None:
            raise ValueError("Model must be created before training")
        if self.train_generator is None:
            raise ValueError("Data generators must be setup before training")
        
        # Create timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"rice_classifier_{timestamp}.h5"
        model_path = os.path.join(self.model_save_dir, model_filename)
        
        # Setup callbacks
        callbacks = self.model.get_callbacks(
            model_save_path=model_path,
            patience=patience
        )
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Model will be saved to: {model_path}")
        
        # Train the model
        self.history = self.model.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate the trained model on test data
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("\nEvaluating model on test data...")
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.model.evaluate(
            self.test_generator,
            verbose=1
        )
        
        # Generate predictions for detailed metrics
        print("Generating predictions...")
        predictions = self.model.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = self.test_generator.classes
        
        # Generate classification report
        report = classification_report(
            true_classes,
            predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Print classification report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(
            true_classes,
            predicted_classes,
            target_names=self.class_names
        ))
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes
        }
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, evaluation_results, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            evaluation_results (dict): Results from evaluate_model()
            save_path (str): Path to save the plot
        """
        # Generate confusion matrix
        cm = confusion_matrix(
            evaluation_results['true_classes'],
            evaluation_results['predicted_classes']
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix - Rice Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()

def main():
    """
    Main training function
    """
    # Configuration
    TRAIN_DIR = "../Data/train"
    TEST_DIR = "../Data/test"
    MODEL_SAVE_DIR = "../models"
    
    # Training parameters
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.2
    PATIENCE = 10
    
    print("Rice Image Classification Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ModelTrainer(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        model_save_dir=MODEL_SAVE_DIR
    )
    
    # Setup data generators
    print("\nSetting up data generators...")
    trainer.setup_data_generators(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        augment_data=True
    )
    
    # Create model
    print("\nCreating model...")
    trainer.create_model(
        freeze_base=True,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train_model(
        epochs=EPOCHS,
        patience=PATIENCE
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = trainer.evaluate_model()
    
    # Create timestamp for saving plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot and save training history
    history_plot_path = os.path.join(MODEL_SAVE_DIR, f"training_history_{timestamp}.png")
    trainer.plot_training_history(save_path=history_plot_path)
    
    # Plot and save confusion matrix
    cm_plot_path = os.path.join(MODEL_SAVE_DIR, f"confusion_matrix_{timestamp}.png")
    trainer.plot_confusion_matrix(evaluation_results, save_path=cm_plot_path)
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Final Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"Final Test Loss: {evaluation_results['test_loss']:.4f}")
    print("\nModel and plots saved to:", MODEL_SAVE_DIR)

if __name__ == "__main__":
    main()

