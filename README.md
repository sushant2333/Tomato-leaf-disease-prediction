# 🍅 Tomato Disease Prediction App

A machine learning application that uses deep learning to predict tomato plant diseases from leaf images. Built with Streamlit and TensorFlow.

## 🌟 Features

- **11 Disease Categories**: Detects 10 different diseases plus healthy plants
- **Real-time Prediction**: Upload images or take photos for instant analysis
- **Detailed Information**: Get disease descriptions, causes, symptoms, and treatment recommendations
- **Random Sample Testing**: Try different disease samples with random selection
- **Beautiful UI**: Modern, responsive interface with interactive visualizations
- **Confidence Metrics**: View prediction confidence with gauge charts

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd tomato_disease_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📦 Deployment Options

### 1. Streamlit Cloud (Recommended)

**Free and Easy Deployment**

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the path to your app: `streamlit_app.py`
   - Click "Deploy"

### 2. Heroku

**For more control and custom domains**

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Add buildpacks**
   ```bash
   heroku buildpacks:add heroku/python
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

### 3. Railway

**Simple deployment with automatic scaling**

1. **Connect to Railway**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway will automatically detect and deploy your Streamlit app

### 4. Docker Deployment

**For containerized deployment**

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 8501
   
   CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**
   ```bash
   docker build -t tomato-disease-app .
   docker run -p 8501:8501 tomato-disease-app
   ```

## 📁 Project Structure

```
tomato_disease_prediction/
├── streamlit_app.py          # Main Streamlit application
├── predict.py               # Prediction model class
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── train/                  # Training dataset
│   ├── healthy/
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   └── ... (other disease folders)
├── valid/                  # Validation dataset
├── advanced_tomato_model.keras  # Trained model
└── README.md
```

## 🧠 Model Information

- **Architecture**: ResNet50V2 with Transfer Learning
- **Input Size**: 224x224 pixels
- **Classes**: 11 (10 diseases + healthy)
- **Framework**: TensorFlow/Keras
- **Accuracy**: High accuracy on tomato disease classification

## 🦠 Supported Diseases

1. **Healthy** - No disease detected
2. **Bacterial Spot** - Bacterial infection
3. **Early Blight** - Fungal disease
4. **Late Blight** - Serious fungal disease
5. **Leaf Mold** - Fungal disease
6. **Septoria Leaf Spot** - Fungal disease
7. **Spider Mites** - Pest infestation
8. **Target Spot** - Fungal disease
9. **Yellow Leaf Curl Virus** - Viral disease
10. **Mosaic Virus** - Viral disease
11. **Powdery Mildew** - Fungal disease

## 🔧 Configuration

### Environment Variables

Create a `.env` file for local development:

```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Model Path

Ensure your trained model file (`advanced_tomato_model.keras`) is in the project root directory.

## 📊 Performance

- **Prediction Time**: < 2 seconds per image
- **Memory Usage**: ~500MB RAM
- **CPU Usage**: Moderate during prediction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: PlantVillage dataset
- Model: ResNet50V2 architecture
- UI: Streamlit framework
- Icons: Emoji icons for better UX

## 📞 Support

If you encounter any issues:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

---

**Happy Tomato Disease Detection! 🍅🔬** 