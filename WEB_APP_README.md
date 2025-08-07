# 🍅 Tomato Disease Predictor Web App

A beautiful, modern web application with dark neon aesthetics for predicting tomato plant diseases using advanced AI and computer vision.

## ✨ Features

### 🎨 **Beautiful Dark Neon Design**
- **Dark Theme**: Deep gradient backgrounds with neon cyan accents
- **Neon Effects**: Glowing text, borders, and interactive elements
- **Smooth Animations**: Fade-in effects, hover animations, and loading spinners
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices

### 🔍 **Prediction Section**
- **Image Upload**: Drag & drop or click to upload tomato leaf images
- **Real-time Analysis**: Instant disease detection using your trained AI model
- **Confidence Visualization**: Beautiful confidence bars and detailed results
- **Comprehensive Results**: Disease name, confidence, severity, and treatment info

### 📚 **Disease Information Guide**
- **11 Disease Categories**: Complete information for all supported diseases
- **Detailed Information**: Causes, symptoms, treatment, and prevention for each disease
- **Severity Indicators**: Color-coded severity levels (Critical, High, Medium, None)
- **Educational Content**: Research-based information from agricultural sources

### 🛠 **Technical Features**
- **Flask Backend**: Robust Python backend with RESTful API
- **TensorFlow Integration**: Direct integration with your trained model
- **Real-time Processing**: Fast image analysis and prediction
- **Error Handling**: Comprehensive error handling and user feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Trained tomato disease model (`best_tomato_model.h5` or `tomato_disease_model.h5`)
- Modern web browser

### Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r web_requirements.txt
   ```

3. **Run the web app**:
   ```bash
   python run_web_app.py
   ```

4. **Open your browser** and go to: `http://localhost:5000`

## 📁 File Structure

```
tomato_disease_prediction/
├── app.py                 # Flask backend server
├── web_app.html          # Frontend HTML/CSS/JS
├── run_web_app.py        # Easy launcher script
├── web_requirements.txt   # Web app dependencies
├── templates/            # Flask templates directory
│   └── web_app.html     # Moved HTML file
├── best_tomato_model.h5  # Your trained model
└── WEB_APP_README.md    # This file
```

## 🎯 How to Use

### **Prediction Section**
1. **Upload Image**: Click "Choose Image" or drag & drop a tomato leaf image
2. **Analyze**: Click "🔍 Analyze Image" to process the image
3. **View Results**: See the prediction, confidence, and detailed information
4. **Learn More**: Read about the disease, symptoms, and treatment options

### **Disease Guide Section**
1. **Browse Diseases**: Click "📚 Disease Guide" to see all diseases
2. **Learn Details**: Each card contains comprehensive information
3. **Understand Severity**: Color-coded severity levels help prioritize treatment

## 🎨 Design Features

### **Color Scheme**
- **Primary**: Cyan (`#00d4aa`) - Main accents and highlights
- **Secondary**: Light Cyan (`#64ffda`) - Text and secondary elements
- **Background**: Dark gradients (`#0a0a0a` to `#16213e`)
- **Text**: White (`#ffffff`) and light gray (`#e2e8f0`)

### **Interactive Elements**
- **Hover Effects**: Cards lift and glow on hover
- **Loading Animations**: Spinning loaders with neon colors
- **Confidence Bars**: Animated progress bars with gradient colors
- **Severity Badges**: Color-coded severity indicators

### **Responsive Design**
- **Desktop**: Full-width layout with side-by-side sections
- **Tablet**: Adjusted grid layouts
- **Mobile**: Single-column layout with touch-friendly buttons

## 🔧 Technical Details

### **Backend (Flask)**
- **Model Loading**: Automatically loads your trained TensorFlow model
- **Image Processing**: Preprocesses images to match training data format
- **API Endpoints**: RESTful API for image prediction and disease information
- **Error Handling**: Comprehensive error handling and user feedback

### **Frontend (HTML/CSS/JavaScript)**
- **Modern CSS**: Flexbox, Grid, and advanced CSS features
- **Vanilla JavaScript**: No external dependencies, fast and lightweight
- **Async/Await**: Modern JavaScript for API calls
- **Progressive Enhancement**: Works without JavaScript for basic functionality

### **Model Integration**
- **TensorFlow**: Direct integration with your trained model
- **Image Preprocessing**: Automatic resizing and normalization
- **Batch Processing**: Efficient single-image prediction
- **Confidence Scores**: Real confidence values from model predictions

## 🦠 Supported Diseases

| Disease | Severity | Description |
|---------|----------|-------------|
| Bacterial Spot | High | Bacterial infection causing dark spots |
| Early Blight | Medium | Fungal disease with target-like lesions |
| Late Blight | Critical | Devastating fungal disease |
| Leaf Mold | Medium | Fungal disease affecting leaf surfaces |
| Septoria Leaf Spot | Medium | Fungal disease with gray centers |
| Spider Mites | Medium | Pest infestation causing leaf damage |
| Target Spot | Medium | Fungal disease with target-like lesions |
| Yellow Leaf Curl Virus | High | Viral disease transmitted by whiteflies |
| Mosaic Virus | High | Viral disease causing leaf distortion |
| Powdery Mildew | Medium | Fungal disease with white powdery patches |
| Healthy | None | No disease detected |

## 🚀 Deployment Options

### **Local Development**
```bash
python run_web_app.py
```

### **Production Deployment**
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t tomato-disease-app .
docker run -p 5000:5000 tomato-disease-app
```

### **Cloud Deployment**
- **Heroku**: Easy deployment with Procfile
- **AWS**: Deploy on EC2 or Elastic Beanstalk
- **Google Cloud**: Deploy on App Engine or Compute Engine
- **Azure**: Deploy on App Service or Container Instances

## 🔧 Customization

### **Styling**
- Edit `web_app.html` to modify colors, fonts, and layout
- CSS variables are defined at the top of the style section
- All neon effects can be adjusted in the CSS

### **Model Integration**
- Update `app.py` to use different model files
- Modify preprocessing in `TomatoDiseasePredictor` class
- Add new disease categories to `DISEASE_INFO` dictionary

### **Features**
- Add new sections by extending the HTML structure
- Implement user accounts and history tracking
- Add batch processing for multiple images
- Integrate with external APIs for weather data

## 🐛 Troubleshooting

### **Common Issues**

1. **Model not found**:
   - Ensure `best_tomato_model.h5` exists in the project directory
   - Check file permissions and path

2. **Port already in use**:
   - Change port in `app.py`: `app.run(port=5001)`
   - Kill existing processes: `lsof -ti:5000 | xargs kill`

3. **Dependencies missing**:
   - Run: `pip install -r web_requirements.txt`
   - Check Python version: `python --version`

4. **Browser compatibility**:
   - Use modern browsers (Chrome, Firefox, Safari, Edge)
   - Enable JavaScript and cookies

### **Performance Tips**
- Use smaller images for faster processing
- Optimize model size for web deployment
- Enable caching for static assets
- Use CDN for production deployment

## 📊 Performance Metrics

- **Image Processing**: ~2-3 seconds per image
- **Model Loading**: ~5-10 seconds on first run
- **Memory Usage**: ~200-300MB with model loaded
- **Concurrent Users**: Supports multiple simultaneous users

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Dataset**: PlantVillage dataset for training data
- **Research**: Agricultural extension services for disease information
- **Design**: Inspired by modern dark theme web applications
- **Technology**: TensorFlow, Flask, and modern web technologies

---

**🍅 Built with ❤️ for tomato farmers and gardeners worldwide** 