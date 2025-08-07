import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def create_data_generators(train_path="train", valid_path="valid", img_size=(224, 224), batch_size=32):
    """Create data generators for training and validation"""
    print("🔄 Creating data generators...")
    
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
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load validation data
    valid_generator = valid_datagen.flow_from_directory(
        valid_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Data generators created!")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {valid_generator.samples}")
    print(f"   Classes: {len(train_generator.class_indices)}")
    
    return train_generator, valid_generator

def build_model(num_classes=11, img_size=(224, 224)):
    """Build the deep learning model"""
    print("🏗️ Building ResNet50V2 model...")
    
    # Use ResNet50V2 as base model
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    
    return model

def train_model(model, train_generator, valid_generator, epochs=20, patience=5):
    """Train the model"""
    print("🚀 Starting model training...")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_tomato_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    print("✅ Training completed!")
    return history

def evaluate_model(model, valid_generator):
    """Evaluate the model performance"""
    print("📊 Evaluating model performance...")
    
    # Get predictions
    predictions = model.predict(valid_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = valid_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Classification report
    class_names = list(valid_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"✅ Evaluation completed!")
    print(f"   Overall Accuracy: {accuracy:.4f}")
    
    return accuracy, report, cm, y_pred, y_true

def plot_training_history(history):
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

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    import seaborn as sns
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

def main():
    """Main function to run the complete training pipeline"""
    print("🍅 Tomato Disease Classification Model Training")
    print("=" * 60)
    
    # Parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 11
    EPOCHS = 20
    PATIENCE = 5
    
    try:
        # Create data generators
        train_generator, valid_generator = create_data_generators(
            img_size=IMG_SIZE, 
            batch_size=BATCH_SIZE
        )
        
        # Build model
        model = build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
        
        # Train model
        history = train_model(model, train_generator, valid_generator, 
                           epochs=EPOCHS, patience=PATIENCE)
        
        # Evaluate model
        accuracy, report, cm, y_pred, y_true = evaluate_model(model, valid_generator)
        
        # Plot results
        plot_training_history(history)
        plot_confusion_matrix(cm, list(valid_generator.class_indices.keys()))
        
        # Save model as .h5
        model.save('tomato_disease_model.h5')
        print("✅ Model saved as 'tomato_disease_model.h5'")
        
        # Print detailed results
        print("\n📈 Detailed Results:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("\nPer-class Performance:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"{class_name}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return model, accuracy
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return None, 0

if __name__ == "__main__":
    model, accuracy = main() 