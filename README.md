# 🍅 Tomato Disease Prediction

A beautiful, AI-powered web application for detecting and diagnosing tomato plant diseases using deep learning. Built with Flask, TensorFlow, and modern web technologies.

![Tomato Disease Predictor](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🔍 AI-Powered Detection**: Advanced deep learning model for accurate disease identification
- **🌙 Dark Neon Theme**: Beautiful dark mode interface with neon accents
- **📱 Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **📚 Comprehensive Disease Guide**: Detailed information about all 11 tomato diseases
- **🎯 Real-time Analysis**: Instant prediction results with confidence scores
- **📊 Visual Results**: Interactive charts and confidence bars
- **🖼️ Sample Images**: Real sample images from the dataset for each disease type

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sushant2333/tomato-disease-prediction.git
   cd tomato-disease-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv tomato_disease_env
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   tomato_disease_env\Scripts\activate
   
   # macOS/Linux
   source tomato_disease_env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r web_requirements.txt
   ```

5. **Download the model**
   - Download the trained model file `best_tomato_model.h5`
   - Place it in the root directory of the project

6. **Run the application**
   ```bash
   python run_web_app.py
   ```

7. **Open your browser**
   - Navigate to `http://localhost:5000`
   - The app will open automatically

## 📁 Project Structure

```
tomato-disease-prediction/
├── app.py                      # Flask backend application
├── run_web_app.py             # Application launcher
├── templates/
│   └── web_app.html          # Main web interface
├── static/
│   └── images/               # Sample disease images
├── web_requirements.txt       # Python dependencies
├── WEB_APP_README.md         # Detailed documentation
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🎯 How to Use

### 1. Disease Prediction
1. Click on the **"🔍 Prediction"** tab
2. Upload a clear image of a tomato leaf
3. Click **"🔍 Analyze Image"**
4. View the prediction results with confidence scores
5. Read detailed information about the detected disease

### 2. Disease Guide
1. Click on the **"📚 Disease Guide"** tab
2. Browse through all 11 tomato diseases
3. View sample images, symptoms, and treatment information
4. Learn about prevention methods for each disease

## 🦠 Supported Diseases

The application can detect and provide information about:

1. **Bacterial Spot** - High severity bacterial disease
2. **Early Blight** - Medium severity fungal disease
3. **Late Blight** - Critical severity fungal disease
4. **Leaf Mold** - Medium severity fungal disease
5. **Septoria Leaf Spot** - Medium severity fungal disease
6. **Spider Mites** - Medium severity pest infestation
7. **Target Spot** - Medium severity fungal disease
8. **Yellow Leaf Curl Virus** - High severity viral disease
9. **Mosaic Virus** - High severity viral disease
10. **Powdery Mildew** - Medium severity fungal disease
11. **Healthy Plants** - No disease detected

## 🛠️ Technical Details

### Backend
- **Framework**: Flask 2.3.3
- **AI Model**: TensorFlow 2.13.0 with custom CNN
- **Image Processing**: Pillow 10.0.0
- **Data Processing**: NumPy 1.24.3

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with gradients and animations
- **JavaScript**: Vanilla JS for interactivity
- **Responsive Design**: Mobile-first approach

### AI Model
- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 pixels
- **Classes**: 11 disease categories
- **Accuracy**: High accuracy on test dataset

## 🎨 Design Features

- **Dark Theme**: Easy on the eyes with neon accents
- **Neon Effects**: Glowing borders and text effects
- **Smooth Animations**: CSS transitions and keyframes
- **Responsive Layout**: Adapts to all screen sizes
- **Interactive Elements**: Hover effects and visual feedback

## 📊 Performance

- **Fast Loading**: Optimized images and assets
- **Real-time Analysis**: Quick prediction results
- **Efficient Model**: Lightweight CNN architecture
- **Caching**: Static file serving for better performance

## 🔧 Customization

### Adding New Diseases
1. Update the `DISEASE_INFO` dictionary in `app.py`
2. Add sample images to `static/images/`
3. Retrain the model with new data
4. Update the class names in the predictor

### Styling Changes
- Modify CSS in `templates/web_app.html`
- Update color schemes and animations
- Customize layout and responsive breakpoints

## 🚀 Deployment

### Local Development
```bash
python run_web_app.py
```

### Production Deployment
1. Use a production WSGI server (Gunicorn, uWSGI)
2. Set up reverse proxy (Nginx, Apache)
3. Configure environment variables
4. Set up SSL certificates

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "run_web_app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset providers for the tomato disease images
- TensorFlow and Keras communities
- Flask development team
- Open source contributors

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/tomato-disease-prediction/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 🔮 Future Enhancements

- [ ] Mobile app version
- [ ] Additional plant species support
- [ ] Real-time camera integration
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] API endpoints for third-party integration

---

**Made with ❤️ for the agricultural community**

*Help farmers protect their crops with AI-powered disease detection!* 
