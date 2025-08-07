import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import glob

# Page configuration
st.set_page_config(
    page_title="🍅 Tomato Disease Predictor",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode styling
st.markdown("""
<style>
    /* Dark mode background and text colors */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
    }
    
    /* Main header with dark theme */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #00d4aa;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(0, 212, 170, 0.5);
    }
    
    /* Sub headers */
    .sub-header {
        font-size: 1.5rem;
        color: #64ffda;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }
    
    /* Prediction box with dark theme */
    .prediction-box {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #00d4aa;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.2);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .prediction-box h3 {
        color: #00d4aa;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(0, 212, 170, 0.5);
    }
    
    .prediction-box p {
        font-size: 1.2rem;
        margin: 0.5rem 0;
        font-weight: 500;
        color: #e2e8f0;
    }
    
    /* Disease info box with dark theme */
    .disease-info {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #64ffda;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(100, 255, 218, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .disease-info h4 {
        color: #64ffda;
        font-size: 1.6rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
    }
    
    .disease-info p {
        font-size: 1.1rem;
        margin: 0.8rem 0;
        line-height: 1.6;
        color: #e2e8f0;
    }
    
    .disease-info strong {
        color: #00d4aa;
        font-weight: bold;
    }
    
    /* Severity indicators with dark theme */
    .severity-critical {
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(229, 62, 62, 0.3);
    }
    
    .severity-high {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
    }
    
    .severity-medium {
        background: linear-gradient(135deg, #ecc94b 0%, #d69e2e 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(236, 201, 75, 0.3);
    }
    
    .severity-none {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(56, 161, 105, 0.3);
    }
    
    /* Button styling with dark theme */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #1a1a2e;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Additional sidebar styling for dark theme */
    .css-1d391kg .css-1d391kg {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Sidebar background */
    .css-1d391kg .css-1d391kg .css-1d391kg {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Sidebar text color */
    .css-1d391kg, .css-1d391kg * {
        color: #ffffff !important;
    }
    
    /* Sidebar headers */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #64ffda !important;
    }
    
    /* Sidebar info boxes */
    .css-1d391kg .stAlert {
        background: rgba(45, 55, 72, 0.9) !important;
        border: 1px solid #64ffda !important;
        color: #ffffff !important;
    }
    
    /* Force sidebar dark background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%) !important;
        color: #ffffff !important;
    }
    
    /* Sidebar content */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #64ffda !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }
    
    /* File uploader styling */
    .stUploadedFile {
        background: rgba(45, 55, 72, 0.8);
        border: 2px solid #00d4aa;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(45, 55, 72, 0.9);
        border: 1px solid #64ffda;
        border-radius: 10px;
    }
    
    /* Success messages */
    .stSuccess {
        background: rgba(56, 161, 105, 0.2);
        border: 1px solid #38a169;
        border-radius: 10px;
    }
    
    /* Error messages */
    .stError {
        background: rgba(229, 62, 62, 0.2);
        border: 1px solid #e53e3e;
        border-radius: 10px;
    }
    
    /* Warning messages */
    .stWarning {
        background: rgba(237, 137, 54, 0.2);
        border: 1px solid #ed8936;
        border-radius: 10px;
    }
    
    /* Info messages */
    .stInfo {
        background: rgba(100, 255, 218, 0.2);
        border: 1px solid #64ffda;
        border-radius: 10px;
    }
    
    /* Plotly charts dark theme */
    .js-plotly-plot {
        background: rgba(45, 55, 72, 0.8) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2d3748;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00d4aa;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00b894;
    }
</style>
""", unsafe_allow_html=True)

# Disease information dictionary
DISEASE_INFO = {
    'healthy': {
        'name': 'Healthy',
        'description': 'No disease detected',
        'cause': 'No disease present',
        'symptoms': 'Green, healthy leaves with normal appearance',
        'treatment': 'Continue regular care and monitoring',
        'severity': 'None',
        'color': '#27ae60'
    },
    'Bacterial_spot': {
        'name': 'Bacterial Spot',
        'description': 'Bacterial infection causing dark spots',
        'cause': 'Xanthomonas spp. bacteria',
        'symptoms': 'Small, dark, water-soaked lesions on leaves',
        'treatment': 'Remove infected plants, use copper-based fungicides',
        'severity': 'High',
        'color': '#e74c3c'
    },
    'Early_blight': {
        'name': 'Early Blight',
        'description': 'Fungal disease with concentric rings',
        'cause': 'Alternaria solani fungus',
        'symptoms': 'Dark brown spots with concentric rings',
        'treatment': 'Remove infected leaves, apply fungicides',
        'severity': 'Medium',
        'color': '#f39c12'
    },
    'Late_blight': {
        'name': 'Late Blight',
        'description': 'Serious fungal disease',
        'cause': 'Phytophthora infestans',
        'symptoms': 'Large, irregular brown lesions with white mold',
        'treatment': 'Immediate removal of infected plants, fungicides',
        'severity': 'Critical',
        'color': '#8e44ad'
    },
    'Leaf_Mold': {
        'name': 'Leaf Mold',
        'description': 'Fungal disease affecting leaf surfaces',
        'cause': 'Passalora fulva fungus',
        'symptoms': 'Yellow spots on upper leaf surface, olive-green mold underneath',
        'treatment': 'Improve air circulation, apply fungicides',
        'severity': 'Medium',
        'color': '#16a085'
    },
    'Septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'description': 'Fungal disease with gray centers',
        'cause': 'Septoria lycopersici fungus',
        'symptoms': 'Small, circular spots with gray centers',
        'treatment': 'Remove infected leaves, apply fungicides',
        'severity': 'Medium',
        'color': '#e67e22'
    },
    'Spider_mites Two-spotted_spider_mite': {
        'name': 'Spider Mites',
        'description': 'Pest infestation causing leaf damage',
        'cause': 'Two-spotted spider mite infestation',
        'symptoms': 'Fine webbing, yellow stippling on leaves',
        'treatment': 'Apply miticides, increase humidity',
        'severity': 'Medium',
        'color': '#d35400'
    },
    'Target_Spot': {
        'name': 'Target Spot',
        'description': 'Fungal disease with target-like lesions',
        'cause': 'Corynespora cassiicola fungus',
        'symptoms': 'Target-like lesions with concentric rings',
        'treatment': 'Remove infected leaves, apply fungicides',
        'severity': 'Medium',
        'color': '#c0392b'
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Yellow Leaf Curl Virus',
        'description': 'Viral disease transmitted by whiteflies',
        'cause': 'Virus transmitted by whiteflies',
        'symptoms': 'Yellow, curled leaves, stunted growth',
        'treatment': 'Remove infected plants, control whiteflies',
        'severity': 'High',
        'color': '#f1c40f'
    },
    'Tomato_mosaic_virus': {
        'name': 'Mosaic Virus',
        'description': 'Viral disease causing leaf distortion',
        'cause': 'Virus transmitted by contact or insects',
        'symptoms': 'Mottled, distorted leaves',
        'treatment': 'Remove infected plants, disinfect tools',
        'severity': 'High',
        'color': '#e67e22'
    },
    'powdery_mildew': {
        'name': 'Powdery Mildew',
        'description': 'Fungal disease with white powdery patches',
        'cause': 'Oidium lycopersicum fungus',
        'symptoms': 'White powdery patches on leaves',
        'treatment': 'Improve air circulation, apply fungicides',
        'severity': 'Medium',
        'color': '#95a5a6'
    }
}

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
                st.success(f"✅ Model loaded from {model_path}")
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                self.model = None
        else:
            st.error(f"❌ Model file {model_path} not found. Please train the model first.")
    
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
            st.error(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict(self, image):
        """Predict disease for a single image"""
        if self.model is None:
            st.error("❌ No model loaded. Please train the model first.")
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
            st.error(f"❌ Error during prediction: {e}")
            return None, None, None

@st.cache_resource
def load_predictor():
    """Load the prediction model (cached for performance)"""
    try:
        predictor = TomatoDiseasePredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence level with dark theme"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)", 'font': {'size': 20, 'color': '#ffffff'}},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#ffffff', 'tickfont': {'color': '#ffffff'}},
            'bar': {'color': "#00d4aa"},
            'steps': [
                {'range': [0, 50], 'color': "#e53e3e"},
                {'range': [50, 80], 'color': "#ed8936"},
                {'range': [80, 100], 'color': "#38a169"}
            ],
            'threshold': {
                'line': {'color': "#00d4aa", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Dark theme layout
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    return fig

def create_prediction_chart(predictions, class_names):
    """Create a bar chart of all predictions with dark theme"""
    # Sort predictions by confidence
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    # Get disease names for display
    disease_names = [DISEASE_INFO.get(cls, {}).get('name', cls) for cls in sorted_classes]
    
    # Dark theme colors for diseases
    dark_colors = {
        'Bacterial_spot': '#e53e3e',
        'Early_blight': '#ed8936', 
        'Late_blight': '#8b5cf6',
        'Leaf_Mold': '#10b981',
        'Septoria_leaf_spot': '#f59e0b',
        'Spider_mites Two-spotted_spider_mite': '#dc2626',
        'Target_Spot': '#7c2d12',
        'Tomato_Yellow_Leaf_Curl_Virus': '#fbbf24',
        'Tomato_mosaic_virus': '#ea580c',
        'healthy': '#059669',
        'powdery_mildew': '#6b7280'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=disease_names,
            y=sorted_predictions * 100,
            marker_color=[dark_colors.get(cls, '#00d4aa') for cls in sorted_classes],
            text=[f'{pred*100:.1f}%' for pred in sorted_predictions],
            textposition='auto',
            textfont={'color': '#ffffff'},
        )
    ])
    
    # Dark theme layout
    fig.update_layout(
        title="All Disease Predictions",
        xaxis_title="Disease Type",
        yaxis_title="Confidence (%)",
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'tickfont': {'color': '#ffffff'}, 'gridcolor': '#4a5568'},
        yaxis={'tickfont': {'color': '#ffffff'}, 'gridcolor': '#4a5568'}
    )
    
    return fig

def get_severity_class(severity):
    """Get CSS class for severity styling"""
    severity_classes = {
        'Critical': 'severity-critical',
        'High': 'severity-high',
        'Medium': 'severity-medium',
        'None': 'severity-none'
    }
    return severity_classes.get(severity, 'severity-medium')

def get_random_disease_sample():
    """Get a random disease sample from the train directory"""
    train_dir = "train"
    if not os.path.exists(train_dir):
        return None
    
    # Get all disease folders
    disease_folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    if not disease_folders:
        return None
    
    # Randomly select a disease folder
    selected_disease = random.choice(disease_folders)
    disease_path = os.path.join(train_dir, selected_disease)
    
    # Get all image files in the selected disease folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(disease_path, ext)))
    
    if not image_files:
        return None
    
    # Randomly select an image
    selected_image = random.choice(image_files)
    return selected_image, selected_disease, len(image_files)

def get_random_healthy_sample():
    """Get a random healthy sample from the train directory"""
    healthy_dir = "train/healthy"
    if not os.path.exists(healthy_dir):
        return None
    
    # Get all image files in the healthy folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(healthy_dir, ext)))
    
    if not image_files:
        return None
    
    # Randomly select an image
    selected_image = random.choice(image_files)
    return selected_image, len(image_files)

def main():
    # Header with dark background
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; border: 2px solid #00d4aa;">
        <h1 class="main-header">🍅 Tomato Disease Predictor</h1>
        <p style="text-align: center; font-size: 1.2rem; color: #64ffda; margin: 0;">Upload a tomato leaf image to identify potential diseases and get treatment recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    if predictor is None:
        st.error("❌ Failed to load the prediction model. Please ensure the model file exists.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📊 Model Information")
    st.sidebar.info("""
    **Model Type**: ResNet50V2 Transfer Learning
    **Classes**: 11 disease categories
    **Input Size**: 224x224 pixels
    **Framework**: TensorFlow/Keras
    """)
    
    st.sidebar.markdown("## 🎯 How to Use")
    st.sidebar.markdown("""
    1. **Upload Image**: Use the file uploader
    2. **Camera Capture**: Take a photo directly
    3. **Sample Images**: Try with provided examples
    4. **View Results**: Get detailed analysis
    """)
    
    st.sidebar.markdown("## 🦠 Supported Diseases")
    disease_list = list(DISEASE_INFO.keys())
    disease_list.remove('healthy')  # Remove healthy from the list since it's not a disease
    disease_names = [DISEASE_INFO[disease]['name'] for disease in disease_list]
    
    for i, disease_name in enumerate(disease_names):
        severity = DISEASE_INFO[disease_list[i]]['severity']
        severity_color = {
            'Critical': '🔴',
            'High': '🟠', 
            'Medium': '🟡',
            'None': '🟢'
        }.get(severity, '🟡')
        st.sidebar.markdown(f"{severity_color} **{disease_name}** ({severity})")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'JPG'],
            help="Upload a clear image of a tomato leaf"
        )
        
        # Camera capture
        camera_photo = st.camera_input("Or take a photo")
        
        # Sample images
        st.markdown("### 🧪 Sample Images")
        sample_col1, sample_col2 = st.columns(2)
        
        with sample_col1:
            if st.button("Try Healthy Sample"):
                result = get_random_healthy_sample()
                if result:
                    sample_path, total_samples = result
                    uploaded_file = open(sample_path, 'rb')
                    st.success(f"✅ Selected random healthy sample: {os.path.basename(sample_path)} (from {total_samples} samples)")
        
        with sample_col2:
            if st.button("Try Disease Sample"):
                result = get_random_disease_sample()
                if result:
                    sample_path, disease_name, total_samples = result
                    uploaded_file = open(sample_path, 'rb')
                    st.success(f"✅ Selected random disease sample: {disease_name} - {os.path.basename(sample_path)} (from {total_samples} samples)")
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Results</h2>', unsafe_allow_html=True)
        
        # Process image and make prediction
        if uploaded_file is not None or camera_photo is not None:
            # Get image from either upload or camera
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_source = "uploaded file"
            else:
                image = Image.open(camera_photo)
                image_source = "camera capture"
            
            # Display original image
            st.image(image, caption=f"Original Image ({image_source})", use_container_width=True)
            
            # Make prediction
            with st.spinner("🔍 Analyzing image..."):
                try:
                    predicted_disease, confidence, predictions = predictor.predict(image)
                    
                    if predicted_disease and predicted_disease in DISEASE_INFO:
                        disease_info = DISEASE_INFO[predicted_disease]
                        severity_class = get_severity_class(disease_info['severity'])
                        
                        # Display prediction results with enhanced styling
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🎯 Prediction Results</h3>
                            <p><strong>Disease:</strong> {disease_info['name']}</p>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                            <div class="{severity_class}">Severity: {disease_info['severity']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence gauge
                        st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
                        
                        # Disease information with enhanced styling
                        st.markdown(f"""
                        <div class="disease-info">
                            <h4>ℹ️ About {disease_info['name']}</h4>
                            <p><strong>Description:</strong> {disease_info['description']}</p>
                            <p><strong>Cause:</strong> {disease_info['cause']}</p>
                            <p><strong>Symptoms:</strong> {disease_info['symptoms']}</p>
                            <p><strong>Treatment:</strong> {disease_info['treatment']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # All predictions chart
                        class_names = predictor.class_names
                        st.plotly_chart(create_prediction_chart(predictions, class_names), use_container_width=True)
                        
                    else:
                        st.error("❌ Could not analyze the image. Please try a different image.")
                        
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
        else:
            st.info("👆 Please upload an image or take a photo to get started!")
    
    # Footer with dark background
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%); text-align: center; color: #64ffda; padding: 2rem; border-radius: 15px; margin-top: 2rem; border: 2px solid #00d4aa;">
        <p style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">🍅 Built with Streamlit & TensorFlow | Tomato Disease Classification Model</p>
        <p style="margin: 0; font-size: 1rem; color: #a0aec0;">For best results, use clear, well-lit images of tomato leaves</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 