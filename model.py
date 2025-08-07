import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TomatoDiseaseClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32, num_classes=11):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def load_and_preprocess_data(self, train_path="train", valid_path="valid"):
        """Load and preprocess the dataset"""
        print("🔄 Loading and preprocessing data...")
        
        # Get class names
        self.class_names = sorted([d for d in os.listdir(train_path) 
                                 if os.path.isdir(os.path.join(train_path, d))])
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Load validation data
        self.valid_generator = valid_datagen.flow_from_directory(
            valid_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Validation samples: {self.valid_generator.samples}")
        print(f"   Classes: {len(self.class_names)}")
        
        return self.train_generator, self.valid_generator
    
    def build_model(self, model_type='resnet'):
        """Build the deep learning model"""
        print(f"🏗️ Building {model_type} model...")
        
        if model_type == 'resnet':
            # Use ResNet50V2 as base model
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif model_type == 'efficientnet':
            # Use EfficientNetB0 as base model
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError("Model type must be 'resnet' or 'efficientnet'")
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        self.model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built successfully!")
        print(f"   Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train_model(self, epochs=50, patience=10):
        """Train the model with callbacks"""
        print("🚀 Starting model training...")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.valid_generator,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
    
    def evaluate_model(self):
        """Evaluate the model performance"""
        print("📊 Evaluating model performance...")
        
        # Get predictions
        predictions = self.model.predict(self.valid_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.valid_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        class_names = list(self.valid_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"✅ Evaluation completed!")
        print(f"   Overall Accuracy: {accuracy:.4f}")
        
        return accuracy, report, cm, y_pred, y_true
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path):
        """Predict disease for a single image"""
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        class_names = list(self.train_generator.class_indices.keys())
        predicted_disease = class_names[predicted_class]
        
        return predicted_disease, confidence, prediction[0]
    
    def save_model(self, filepath='tomato_disease_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='tomato_disease_model.h5'):
        """Load a trained model"""
        self.model = models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")

def main():
    """Main function to run the complete pipeline"""
    print("🍅 Tomato Disease Classification Model")
    print("=" * 50)
    
    # Initialize the classifier
    classifier = TomatoDiseaseClassifier(
        img_size=(224, 224),
        batch_size=32,
        num_classes=11
    )
    
    # Load and preprocess data
    train_gen, valid_gen = classifier.load_and_preprocess_data()
    
    # Build model
    model = classifier.build_model(model_type='resnet')
    
    # Train model
    history = classifier.train_model(epochs=30, patience=8)
    
    # Evaluate model
    accuracy, report, cm, y_pred, y_true = classifier.evaluate_model()
    
    # Plot results
    classifier.plot_training_history(history)
    classifier.plot_confusion_matrix(cm, list(valid_gen.class_indices.keys()))
    
    # Save model
    classifier.save_model()
    
    # Print detailed results
    print("\n📈 Detailed Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nPer-class Performance:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{class_name}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    return classifier

if __name__ == "__main__":
    classifier = main() 