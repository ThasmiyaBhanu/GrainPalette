import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

class RiceClassifierModel:
    """
    Rice image classification model using MobileNet as feature extractor
    """
    
    def __init__(self, num_classes=5, input_shape=(224, 224, 3), weights='imagenet'):
        """
        Initialize the rice classifier model
        
        Args:
            num_classes (int): Number of rice varieties to classify
            input_shape (tuple): Input image shape
            weights (str): Pre-trained weights to use
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weights = weights
        self.model = None
        
    def build_model(self, freeze_base=True, dropout_rate=0.2):
        """
        Build the MobileNet-based classification model
        
        Args:
            freeze_base (bool): Whether to freeze the base MobileNet layers
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            tf.keras.Model: Compiled model
        """
        # Load pre-trained MobileNet without top layers
        base_model = MobileNet(
            weights=self.weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers if specified
        if freeze_base:
            base_model.trainable = False
            print(f"Base model frozen. Trainable parameters: {base_model.count_params()}")
        else:
            base_model.trainable = True
            print(f"Base model unfrozen. Trainable parameters: {base_model.count_params()}")
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, activation='relu', name='dense_2')(x)
        x = Dropout(dropout_rate)(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, metrics=['accuracy']):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate (float): Learning rate for optimizer
            metrics (list): Metrics to track during training
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        print("Model compiled successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        # Count trainable parameters
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable_params:,}")
        
    def get_callbacks(self, model_save_path, patience=10, min_lr=1e-7):
        """
        Get training callbacks
        
        Args:
            model_save_path (str): Path to save the best model
            patience (int): Patience for early stopping
            min_lr (float): Minimum learning rate
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=min_lr,
                verbose=1
            )
        ]
        
        return callbacks
    
    def print_model_summary(self):
        """
        Print model architecture summary
        """
        if self.model is None:
            raise ValueError("Model must be built before printing summary")
            
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*50)
        self.model.summary()
        print("="*50 + "\n")
    
    def save_model_architecture(self, filepath):
        """
        Save model architecture to JSON file
        
        Args:
            filepath (str): Path to save the model architecture
        """
        if self.model is None:
            raise ValueError("Model must be built before saving architecture")
            
        model_json = self.model.to_json()
        with open(filepath, 'w') as json_file:
            json_file.write(model_json)
        print(f"Model architecture saved to: {filepath}")

def create_rice_classifier(num_classes=5, input_shape=(224, 224, 3), 
                          freeze_base=True, dropout_rate=0.2, learning_rate=0.001):
    """
    Convenience function to create and compile a rice classifier model
    
    Args:
        num_classes (int): Number of rice varieties
        input_shape (tuple): Input image shape
        freeze_base (bool): Whether to freeze base model
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate
        
    Returns:
        RiceClassifierModel: Compiled model instance
    """
    # Create model instance
    classifier = RiceClassifierModel(num_classes=num_classes, input_shape=input_shape)
    
    # Build the model
    model = classifier.build_model(freeze_base=freeze_base, dropout_rate=dropout_rate)
    
    # Compile the model
    classifier.compile_model(learning_rate=learning_rate)
    
    # Print summary
    classifier.print_model_summary()
    
    return classifier

if __name__ == "__main__":
    # Example usage
    print("Creating Rice Classification Model...")
    
    # Create the model
    rice_model = create_rice_classifier(
        num_classes=5,
        input_shape=(224, 224, 3),
        freeze_base=True,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # Save model architecture
    os.makedirs('../models', exist_ok=True)
    rice_model.save_model_architecture('../models/rice_classifier_architecture.json')
    
    print("\nModel created successfully!")
    print("Ready for training with data generators.")

