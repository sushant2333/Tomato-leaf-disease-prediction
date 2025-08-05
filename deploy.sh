#!/bin/bash

# 🍅 Tomato Disease Prediction App - Deployment Script
# This script helps you deploy your Streamlit app to various platforms

echo "🍅 Tomato Disease Prediction App - Deployment Helper"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to deploy to Streamlit Cloud
deploy_streamlit_cloud() {
    echo "🚀 Deploying to Streamlit Cloud..."
    
    # Check if git is installed
    if ! command_exists git; then
        echo "❌ Git is not installed. Please install Git first."
        return 1
    fi
    
    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        echo "❌ Not in a git repository. Please initialize git first:"
        echo "   git init"
        echo "   git add ."
        echo "   git commit -m 'Initial commit'"
        return 1
    fi
    
    echo "✅ Git repository found"
    echo "📝 Steps to deploy:"
    echo "1. Push your code to GitHub:"
    echo "   git remote add origin <your-github-repo-url>"
    echo "   git push -u origin main"
    echo ""
    echo "2. Go to https://share.streamlit.io"
    echo "3. Sign in with GitHub"
    echo "4. Click 'New app'"
    echo "5. Select your repository"
    echo "6. Set path to: streamlit_app.py"
    echo "7. Click 'Deploy'"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "🚀 Deploying to Heroku..."
    
    if ! command_exists heroku; then
        echo "❌ Heroku CLI is not installed."
        echo "📥 Install it from: https://devcenter.heroku.com/articles/heroku-cli"
        return 1
    fi
    
    # Check if logged in to Heroku
    if ! heroku auth:whoami >/dev/null 2>&1; then
        echo "❌ Not logged in to Heroku. Please run: heroku login"
        return 1
    fi
    
    # Create Heroku app
    echo "📦 Creating Heroku app..."
    heroku create tomato-disease-predictor-$(date +%s)
    
    # Add buildpacks
    echo "🔧 Adding buildpacks..."
    heroku buildpacks:add heroku/python
    
    # Deploy
    echo "🚀 Deploying..."
    git push heroku main
    
    echo "✅ Deployment complete!"
    echo "🌐 Your app is available at: $(heroku info -s | grep web_url | cut -d= -f2)"
}

# Function to deploy with Docker
deploy_docker() {
    echo "🐳 Deploying with Docker..."
    
    if ! command_exists docker; then
        echo "❌ Docker is not installed. Please install Docker first."
        return 1
    fi
    
    # Build Docker image
    echo "🔨 Building Docker image..."
    docker build -t tomato-disease-app .
    
    # Run container
    echo "🚀 Starting container..."
    docker run -d -p 8501:8501 --name tomato-disease-app tomato-disease-app
    
    echo "✅ Docker deployment complete!"
    echo "🌐 Your app is available at: http://localhost:8501"
    echo "🛑 To stop: docker stop tomato-disease-app"
    echo "🗑️  To remove: docker rm tomato-disease-app"
}

# Function to deploy to Railway
deploy_railway() {
    echo "🚂 Deploying to Railway..."
    
    if ! command_exists railway; then
        echo "❌ Railway CLI is not installed."
        echo "📥 Install it with: npm install -g @railway/cli"
        return 1
    fi
    
    # Login to Railway
    echo "🔐 Logging in to Railway..."
    railway login
    
    # Deploy
    echo "🚀 Deploying..."
    railway up
    
    echo "✅ Railway deployment complete!"
}

# Function to check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check Python
    if ! command_exists python3; then
        echo "❌ Python 3 is not installed"
        return 1
    fi
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        echo "❌ requirements.txt not found"
        return 1
    fi
    
    # Check if streamlit_app.py exists
    if [ ! -f "streamlit_app.py" ]; then
        echo "❌ streamlit_app.py not found"
        return 1
    fi
    
    echo "✅ All prerequisites met!"
    return 0
}

# Function to test locally
test_local() {
    echo "🧪 Testing locally..."
    
    if ! check_prerequisites; then
        return 1
    fi
    
    # Install dependencies
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    
    # Run the app
    echo "🚀 Starting Streamlit app..."
    echo "🌐 Open http://localhost:8501 in your browser"
    echo "🛑 Press Ctrl+C to stop"
    
    streamlit run streamlit_app.py
}

# Main menu
show_menu() {
    echo ""
    echo "Choose deployment option:"
    echo "1) Test locally"
    echo "2) Deploy to Streamlit Cloud (Recommended)"
    echo "3) Deploy to Heroku"
    echo "4) Deploy with Docker"
    echo "5) Deploy to Railway"
    echo "6) Check prerequisites"
    echo "7) Exit"
    echo ""
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1) test_local ;;
        2) deploy_streamlit_cloud ;;
        3) deploy_heroku ;;
        4) deploy_docker ;;
        5) deploy_railway ;;
        6) check_prerequisites ;;
        7) echo "👋 Goodbye!"; exit 0 ;;
        *) echo "❌ Invalid choice. Please try again." ;;
    esac
}

# Show menu
show_menu 