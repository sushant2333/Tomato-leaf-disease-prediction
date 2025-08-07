from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json

app = Flask(__name__, static_folder='static')

# Disease information (same as in the HTML)
DISEASE_INFO = {
    'Bacterial_spot': {
        'name': 'Bacterial Spot',
        'description': 'A serious bacterial disease that affects tomato plants, causing dark spots on leaves, stems, and fruits.',
        'cause': 'Caused by Xanthomonas spp. bacteria, particularly Xanthomonas vesicatoria. Spread through contaminated seeds, plant debris, and water splash.',
        'symptoms': 'Small, dark, water-soaked lesions on leaves that become brown with yellow halos. Lesions may coalesce and cause defoliation. Similar spots appear on stems and fruits.',
        'treatment': 'Remove and destroy infected plants. Use copper-based bactericides. Practice crop rotation. Use disease-free seeds. Maintain proper spacing for air circulation.',
        'prevention': 'Use resistant varieties. Avoid overhead irrigation. Sanitize tools between plants. Remove plant debris after harvest.',
        'severity': 'High',
        'color': '#e53e3e'
    },
    'Early_blight': {
        'name': 'Early Blight',
        'description': 'A common fungal disease that causes characteristic target-like lesions on tomato leaves.',
        'cause': 'Caused by Alternaria solani fungus. Favored by warm, humid conditions and stressed plants.',
        'symptoms': 'Dark brown spots with concentric rings (target-like appearance) on lower leaves first. Lesions may have yellow halos. Severe infection causes defoliation.',
        'treatment': 'Remove infected leaves. Apply fungicides containing chlorothalonil or mancozeb. Improve air circulation. Mulch to prevent soil splash.',
        'prevention': 'Use resistant varieties. Space plants properly. Avoid overhead irrigation. Remove lower leaves as plant grows.',
        'severity': 'Medium',
        'color': '#ed8936'
    },
    'Late_blight': {
        'name': 'Late Blight',
        'description': 'A devastating fungal disease that can rapidly destroy entire tomato crops.',
        'cause': 'Caused by Phytophthora infestans. Thrives in cool, wet conditions. Spreads rapidly through wind and water.',
        'symptoms': 'Large, irregular brown lesions on leaves with white mold on undersides. Lesions may appear water-soaked. Rapid defoliation and fruit rot.',
        'treatment': 'Immediate removal of infected plants. Apply fungicides containing copper or chlorothalonil. Improve drainage and air circulation.',
        'prevention': 'Use resistant varieties. Avoid overhead irrigation. Space plants widely. Monitor weather conditions. Apply preventive fungicides.',
        'severity': 'Critical',
        'color': '#8b5cf6'
    },
    'Leaf_Mold': {
        'name': 'Leaf Mold',
        'description': 'A fungal disease that primarily affects greenhouse-grown tomatoes but can occur outdoors.',
        'cause': 'Caused by Passalora fulva (formerly Cladosporium fulvum). Favored by high humidity and poor air circulation.',
        'symptoms': 'Yellow spots on upper leaf surface with olive-green to grayish-brown mold on lower surface. Lesions may coalesce and cause defoliation.',
        'treatment': 'Improve air circulation and reduce humidity. Apply fungicides containing chlorothalonil or copper. Remove infected leaves.',
        'prevention': 'Use resistant varieties. Maintain proper spacing. Ventilate greenhouses. Avoid overhead irrigation.',
        'severity': 'Medium',
        'color': '#10b981'
    },
    'Septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'description': 'A fungal disease that causes small, circular spots on tomato leaves.',
        'cause': 'Caused by Septoria lycopersici. Spreads through water splash and contaminated tools.',
        'symptoms': 'Small, circular spots with gray centers and dark borders. Spots may have black dots (fungal spores) in centers. Severe infection causes defoliation.',
        'treatment': 'Remove infected leaves. Apply fungicides containing chlorothalonil or mancozeb. Improve air circulation.',
        'prevention': 'Use disease-free seeds. Space plants properly. Avoid overhead irrigation. Remove plant debris.',
        'severity': 'Medium',
        'color': '#f59e0b'
    },
    'Spider_mites Two-spotted_spider_mite': {
        'name': 'Spider Mites',
        'description': 'Tiny arachnids that feed on plant sap, causing stippling and webbing on leaves.',
        'cause': 'Two-spotted spider mites (Tetranychus urticae). Thrive in hot, dry conditions. Spread through wind and contaminated plants.',
        'symptoms': 'Fine stippling (tiny white dots) on leaves. Fine webbing on leaf undersides. Leaves may turn yellow and drop. Severe infestation causes defoliation.',
        'treatment': 'Apply miticides or insecticidal soaps. Use predatory mites. Increase humidity. Remove heavily infested leaves.',
        'prevention': 'Monitor plants regularly. Maintain adequate humidity. Avoid over-fertilization. Use reflective mulches.',
        'severity': 'Medium',
        'color': '#dc2626'
    },
    'Target_Spot': {
        'name': 'Target Spot',
        'description': 'A fungal disease that causes target-like lesions on leaves and fruits.',
        'cause': 'Caused by Corynespora cassiicola. Spreads through water splash and contaminated tools.',
        'symptoms': 'Target-like lesions with concentric rings on leaves. Lesions may have yellow halos. Similar spots on stems and fruits.',
        'treatment': 'Remove infected leaves. Apply fungicides containing chlorothalonil or mancozeb. Improve air circulation.',
        'prevention': 'Use resistant varieties. Space plants properly. Avoid overhead irrigation. Remove plant debris.',
        'severity': 'Medium',
        'color': '#7c2d12'
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Yellow Leaf Curl Virus',
        'description': 'A viral disease transmitted by whiteflies that causes severe stunting and leaf curling.',
        'cause': 'Transmitted by whiteflies (Bemisia tabaci). Virus can persist in weeds and other host plants.',
        'symptoms': 'Yellow, upward-curled leaves. Stunted growth. Reduced fruit production. Plants may appear bushy due to shortened internodes.',
        'treatment': 'Remove infected plants immediately. Control whiteflies with insecticides or biological controls. No cure for infected plants.',
        'prevention': 'Use resistant varieties. Control whiteflies. Remove weeds and alternative hosts. Use reflective mulches.',
        'severity': 'High',
        'color': '#fbbf24'
    },
    'Tomato_mosaic_virus': {
        'name': 'Mosaic Virus',
        'description': 'A viral disease that causes mottled, distorted leaves and reduced fruit quality.',
        'cause': 'Transmitted by contact, contaminated tools, and some insects. Can persist in plant debris and seeds.',
        'symptoms': 'Mottled, distorted leaves with light and dark green areas. Stunted growth. Reduced fruit production and quality.',
        'treatment': 'Remove infected plants. Disinfect tools between plants. No cure for infected plants.',
        'prevention': 'Use disease-free seeds. Disinfect tools. Control insects. Remove plant debris. Avoid handling plants when wet.',
        'severity': 'High',
        'color': '#ea580c'
    },
    'powdery_mildew': {
        'name': 'Powdery Mildew',
        'description': 'A fungal disease that creates white powdery patches on leaves and stems.',
        'cause': 'Caused by Oidium lycopersicum. Favored by high humidity and moderate temperatures.',
        'symptoms': 'White to grayish powdery patches on leaves, stems, and sometimes fruits. Leaves may become distorted and drop.',
        'treatment': 'Apply fungicides containing sulfur or potassium bicarbonate. Improve air circulation. Remove infected leaves.',
        'prevention': 'Use resistant varieties. Maintain proper spacing. Avoid overhead irrigation. Monitor humidity levels.',
        'severity': 'Medium',
        'color': '#6b7280'
    },
    'healthy': {
        'name': 'Healthy Plant',
        'description': 'A healthy tomato plant with no signs of disease or pest infestation.',
        'cause': 'No disease present. Plant is growing under optimal conditions.',
        'symptoms': 'Green, healthy leaves with normal appearance. Good growth and development. No spots, lesions, or discoloration.',
        'treatment': 'Continue regular care and monitoring. Maintain optimal growing conditions.',
        'prevention': 'Practice good cultural practices. Monitor plants regularly. Use disease-resistant varieties.',
        'severity': 'None',
        'color': '#059669'
    }
}

class TomatoDiseasePredictor:
    def __init__(self, model_path='best_tomato_model.h5', img_size=(224, 224)):
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
            print(f"❌ Model file {model_path} not found.")
    
    def preprocess_image(self, image):
        """Preprocess a single image for prediction"""
        try:
            # Convert to RGB if it's not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.img_size)
            
            # Convert to array and normalize
            img_array = img_to_array(image)
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict(self, image):
        """Predict disease for a single image"""
        if self.model is None:
            return None, None, None
        
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        if img_array is None:
            return None, None, None
        
        try:
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Get predicted disease name
            predicted_disease = self.class_names[predicted_class]
            
            return predicted_disease, confidence, predictions[0]
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None, None, None

# Initialize predictor
predictor = TomatoDiseasePredictor()

@app.route('/')
def index():
    return render_template('web_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Remove data URL prefix
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        predicted_disease, confidence, predictions = predictor.predict(image)
        
        if predicted_disease and predicted_disease in DISEASE_INFO:
            disease_info = DISEASE_INFO[predicted_disease]
            
            # Get all predictions for chart
            all_predictions = {}
            for i, class_name in enumerate(predictor.class_names):
                all_predictions[class_name] = float(predictions[i])
            
            response = {
                'success': True,
                'prediction': {
                    'disease': predicted_disease,
                    'name': disease_info['name'],
                    'confidence': float(confidence),
                    'severity': disease_info['severity'],
                    'description': disease_info['description'],
                    'cause': disease_info['cause'],
                    'symptoms': disease_info['symptoms'],
                    'treatment': disease_info['treatment'],
                    'prevention': disease_info['prevention'],
                    'color': disease_info['color']
                },
                'all_predictions': all_predictions
            }
        else:
            response = {
                'success': False,
                'error': 'Could not analyze the image. Please try a different image.'
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error during prediction: {str(e)}'
        })

@app.route('/diseases')
def get_diseases():
    """Get all disease information"""
    return jsonify(DISEASE_INFO)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Move the HTML file to templates directory
    if os.path.exists('web_app.html'):
        import shutil
        shutil.move('web_app.html', 'templates/web_app.html')
    
    print("🍅 Starting Tomato Disease Predictor Web App...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 