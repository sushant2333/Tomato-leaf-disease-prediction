import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from predict import TomatoDiseasePredictor
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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #e74c3c;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #e74c3c;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
    }
    .prediction-box h3 {
        color: white;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-box p {
        font-size: 1.2rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .disease-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #e74c3c;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .disease-info h4 {
        color: white;
        font-size: 1.6rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .disease-info p {
        font-size: 1.1rem;
        margin: 0.8rem 0;
        line-height: 1.6;
    }
    .disease-info strong {
        color: #f1c40f;
        font-weight: bold;
    }
    .confidence-bar {
        background-color: #ecf0f1;
        border-radius: 10px;
        overflow: hidden;
        height: 25px;
        margin: 0.5rem 0;
        border: 2px solid #bdc3c7;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #27ae60, #2ecc71);
        transition: width 0.5s ease;
    }
    .upload-section {
        border: 3px dashed #e74c3c;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.8rem 0;
        border: 2px solid #e74c3c;
    }
    .severity-critical {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .severity-high {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .severity-medium {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .severity-none {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .stFileUploader > div > div > div {
        border: 3px dashed #e74c3c;
        border-radius: 15px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stCameraInput > div {
        border: 3px solid #e74c3c;
        border-radius: 15px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the prediction model (cached for performance)"""
    try:
        predictor = TomatoDiseasePredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence level"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)", 'font': {'size': 20}},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_prediction_chart(predictions, class_names):
    """Create a bar chart of all predictions"""
    # Sort predictions by confidence
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    # Get disease names for display
    disease_names = [DISEASE_INFO.get(cls, {}).get('name', cls) for cls in sorted_classes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=disease_names,
            y=sorted_predictions * 100,
            marker_color=[DISEASE_INFO.get(cls, {}).get('color', '#3498db') for cls in sorted_classes],
            text=[f'{pred*100:.1f}%' for pred in sorted_predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="All Disease Predictions",
        xaxis_title="Disease Type",
        yaxis_title="Confidence (%)",
        height=400,
        showlegend=False
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
    # Header
    st.markdown('<h1 class="main-header">🍅 Tomato Disease Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Upload a tomato leaf image to identify potential diseases and get treatment recommendations</p>', unsafe_allow_html=True)
    
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
        
        # Clear session state when user uploads their own file
        if uploaded_file is not None and (not hasattr(st.session_state, 'sample_path') or st.session_state.sample_path != uploaded_file.name):
            if hasattr(st.session_state, 'selected_sample'):
                del st.session_state.selected_sample
            if hasattr(st.session_state, 'sample_path'):
                del st.session_state.sample_path
        
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
                    st.session_state.selected_sample = "Healthy"
                    st.session_state.sample_path = sample_path
                    st.success(f"✅ Selected random healthy sample: {os.path.basename(sample_path)} (from {total_samples} samples)")
        
        with sample_col2:
            if st.button("Try Disease Sample"):
                result = get_random_disease_sample()
                if result:
                    sample_path, disease_name, total_samples = result
                    uploaded_file = open(sample_path, 'rb')
                    st.session_state.selected_sample = disease_name
                    st.session_state.sample_path = sample_path
                    st.success(f"✅ Selected random disease sample: {disease_name} - {os.path.basename(sample_path)} (from {total_samples} samples)")
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Results</h2>', unsafe_allow_html=True)
        
        # Process image and make prediction
        if uploaded_file is not None or camera_photo is not None:
            # Get image from either upload or camera
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_source = "uploaded file"
                
                # Show sample information if it's a sample
                if hasattr(st.session_state, 'selected_sample') and hasattr(st.session_state, 'sample_path'):
                    if st.session_state.selected_sample == "Healthy":
                        st.info(f"🍃 Sample: Healthy tomato leaf - {os.path.basename(st.session_state.sample_path)}")
                    else:
                        st.info(f"🦠 Sample: {st.session_state.selected_sample} - {os.path.basename(st.session_state.sample_path)}")
            else:
                image = Image.open(camera_photo)
                image_source = "camera capture"
            
            # Display original image
            st.image(image, caption=f"Original Image ({image_source})", use_container_width=True)
            
            # Save image temporarily for prediction
            temp_path = "temp_streamlit_image.jpg"
            image.save(temp_path)
            
            # Make prediction
            with st.spinner("🔍 Analyzing image..."):
                try:
                    predicted_disease, confidence, predictions, original_img = predictor.predict(temp_path)
                    
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
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.info("👆 Please upload an image or take a photo to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p>🍅 Built with Streamlit & TensorFlow | Tomato Disease Classification Model</p>
        <p>For best results, use clear, well-lit images of tomato leaves</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 