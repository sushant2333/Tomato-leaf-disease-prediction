#!/usr/bin/env python3
"""
Tomato Disease Predictor Web App Launcher
This script launches the beautiful dark neon web app for tomato disease prediction.
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'tensorflow', 
        'pillow',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    
    model_files = ['best_tomato_model.h5', 'tomato_disease_model.h5']
    found_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found model: {model_file}")
            found_models.append(model_file)
        else:
            print(f"❌ Model not found: {model_file}")
    
    if not found_models:
        print("\n⚠️  No model files found!")
        print("Please ensure you have trained the model first.")
        print("Run 'python train_model.py' to train the model.")
        return False
    
    return True

def create_templates_directory():
    """Create templates directory if it doesn't exist"""
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created templates directory")

def move_html_to_templates():
    """Move HTML file to templates directory"""
    if os.path.exists('web_app.html') and not os.path.exists('templates/web_app.html'):
        import shutil
        shutil.move('web_app.html', 'templates/web_app.html')
        print("📄 Moved web_app.html to templates directory")

def main():
    """Main launcher function"""
    print("🍅 Tomato Disease Predictor Web App")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model files
    if not check_model_files():
        return
    
    # Setup directories
    create_templates_directory()
    move_html_to_templates()
    
    print("\n🚀 Starting web app...")
    print("🌐 The app will open in your browser automatically")
    print("📱 You can also manually visit: http://localhost:5000")
    print("\n" + "=" * 50)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            print("⚠️  Could not open browser automatically")
            print("Please manually visit: http://localhost:5000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Web app stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting web app: {e}")
        print("Please check if port 5000 is available")

if __name__ == "__main__":
    main() 