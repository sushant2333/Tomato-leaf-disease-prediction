import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import glob

class TomatoDiseasePredictor:
    def __init__(self, model_path='tomato_disease_model.h5', img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        self.class_names = [
            'Bacterial_spot',
            'Early_blight', 
            'Late_blight',
            'Leaf_Mold',
            'Septoria_leaf_spot',
            'Spider_mites Two-spotted_spider_mite',
            'Target_Spot',
            'Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato_mosaic_virus',
            'healthy',
            'powdery_mildew'
        ]
        
        # Load model if it exists
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.model = None
        else:
            print(f"❌ Model file {model_path} not found. Please train the model first.")
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Resize image
            img = img.resize(self.img_size)
            
            # Convert to array and normalize
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, img
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None, None
    
    def predict(self, image_path):
        """Predict disease for a single image"""
        if self.model is None:
            print("❌ No model loaded. Please train the model first.")
            return None, None, None
        
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        
        if img_array is None:
            return None, None, None
        
        try:
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Get predicted disease name
            predicted_disease = self.class_names[predicted_class]
            
            return predicted_disease, confidence, predictions[0], original_img
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None, None, None, None
    
    def predict_batch(self, image_folder):
        """Predict diseases for all images in a folder"""
        results = []
        
        if not os.path.exists(image_folder):
            print(f"❌ Folder {image_folder} not found.")
            return results
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        
        print(f"🔍 Found {len(image_files)} images to predict...")
        
        for image_path in image_files:
            predicted_disease, confidence, predictions, original_img = self.predict(image_path)
            
            if predicted_disease is not None:
                results.append({
                    'image_path': image_path,
                    'predicted_disease': predicted_disease,
                    'confidence': confidence,
                    'predictions': predictions
                })
        
        return results
    
    def display_prediction(self, image_path, save_plot=True):
        """Display prediction results with visualization"""
        predicted_disease, confidence, predictions, original_img = self.predict(image_path)
        
        if predicted_disease is None:
            return
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display original image
        ax1.imshow(original_img)
        ax1.set_title(f'Original Image\n{os.path.basename(image_path)}')
        ax1.axis('off')
        
        # Display prediction results
        if predictions is not None:
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[::-1][:3]
            
            y_pos = np.arange(3)
            top_diseases = [self.class_names[i] for i in top_indices]
            top_confidences = [predictions[i] for i in top_indices]
            
            bars = ax2.barh(y_pos, top_confidences)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_diseases)
            ax2.set_xlabel('Confidence')
            ax2.set_title('Top 3 Predictions')
            
            # Color the bars
            colors = ['green' if i == predicted_disease else 'lightblue' for i in top_diseases]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add confidence values on bars
            for i, (bar, conf) in enumerate(zip(bars, top_confidences)):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{conf:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print results
        print(f"\n🍅 Prediction Results:")
        print(f"   Image: {os.path.basename(image_path)}")
        print(f"   Predicted Disease: {predicted_disease}")
        print(f"   Confidence: {confidence:.4f}")
        
        return predicted_disease, confidence

def main():
    """Main function to test predictions"""
    print("🍅 Tomato Disease Prediction")
    print("=" * 40)
    
    # Initialize predictor
    predictor = TomatoDiseasePredictor()
    
    if predictor.model is None:
        print("❌ No trained model found. Please run train_model.py first.")
        return
    
    # Test prediction on a sample image
    print("\n🧪 Testing prediction on sample images...")
    
    # Find a sample image from the training data
    sample_image = None
    for class_name in predictor.class_names:
        class_path = os.path.join("train", class_name)
        if os.path.exists(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_image = os.path.join(class_path, file)
                    break
            if sample_image:
                break
    
    if sample_image:
        print(f"Testing on: {os.path.basename(sample_image)}")
        predictor.display_prediction(sample_image)
    else:
        print("❌ No sample images found in training data.")
    
    # Interactive prediction
    print("\n🔍 Interactive Prediction:")
    print("Enter the path to an image file (or 'quit' to exit):")
    
    while True:
        image_path = input("Image path: ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print("❌ File not found. Please enter a valid path.")
            continue
        
        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print("❌ Please enter a valid image file (.jpg, .jpeg, .png)")
            continue
        
        predictor.display_prediction(image_path)

if __name__ == "__main__":
    main() 