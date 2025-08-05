@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🍅 Tomato Disease Prediction App - Deployment Helper
echo ==================================================

:menu
echo.
echo Choose deployment option:
echo 1) Test locally
echo 2) Deploy to Streamlit Cloud (Recommended)
echo 3) Deploy to Heroku
echo 4) Deploy with Docker
echo 5) Deploy to Railway
echo 6) Check prerequisites
echo 7) Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto test_local
if "%choice%"=="2" goto deploy_streamlit_cloud
if "%choice%"=="3" goto deploy_heroku
if "%choice%"=="4" goto deploy_docker
if "%choice%"=="5" goto deploy_railway
if "%choice%"=="6" goto check_prerequisites
if "%choice%"=="7" goto exit
echo ❌ Invalid choice. Please try again.
goto menu

:test_local
echo 🧪 Testing locally...
call :check_prerequisites
if errorlevel 1 goto menu

echo 📦 Installing dependencies...
pip install -r requirements.txt

echo 🚀 Starting Streamlit app...
echo 🌐 Open http://localhost:8501 in your browser
echo 🛑 Press Ctrl+C to stop
streamlit run streamlit_app.py
goto menu

:deploy_streamlit_cloud
echo 🚀 Deploying to Streamlit Cloud...

where git >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install Git first.
    goto menu
)

if not exist ".git" (
    echo ❌ Not in a git repository. Please initialize git first:
    echo    git init
    echo    git add .
    echo    git commit -m "Initial commit"
    goto menu
)

echo ✅ Git repository found
echo 📝 Steps to deploy:
echo 1. Push your code to GitHub:
echo    git remote add origin ^<your-github-repo-url^>
echo    git push -u origin main
echo.
echo 2. Go to https://share.streamlit.io
echo 3. Sign in with GitHub
echo 4. Click "New app"
echo 5. Select your repository
echo 6. Set path to: streamlit_app.py
echo 7. Click "Deploy"
goto menu

:deploy_heroku
echo 🚀 Deploying to Heroku...

where heroku >nul 2>&1
if errorlevel 1 (
    echo ❌ Heroku CLI is not installed.
    echo 📥 Install it from: https://devcenter.heroku.com/articles/heroku-cli
    goto menu
)

heroku auth:whoami >nul 2>&1
if errorlevel 1 (
    echo ❌ Not logged in to Heroku. Please run: heroku login
    goto menu
)

echo 📦 Creating Heroku app...
for /f "tokens=*" %%i in ('heroku create tomato-disease-predictor-%random%') do set app_name=%%i

echo 🔧 Adding buildpacks...
heroku buildpacks:add heroku/python

echo 🚀 Deploying...
git push heroku main

echo ✅ Deployment complete!
echo 🌐 Your app is available at: %app_name%
goto menu

:deploy_docker
echo 🐳 Deploying with Docker...

where docker >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker first.
    goto menu
)

echo 🔨 Building Docker image...
docker build -t tomato-disease-app .

echo 🚀 Starting container...
docker run -d -p 8501:8501 --name tomato-disease-app tomato-disease-app

echo ✅ Docker deployment complete!
echo 🌐 Your app is available at: http://localhost:8501
echo 🛑 To stop: docker stop tomato-disease-app
echo 🗑️  To remove: docker rm tomato-disease-app
goto menu

:deploy_railway
echo 🚂 Deploying to Railway...

where railway >nul 2>&1
if errorlevel 1 (
    echo ❌ Railway CLI is not installed.
    echo 📥 Install it with: npm install -g @railway/cli
    goto menu
)

echo 🔐 Logging in to Railway...
railway login

echo 🚀 Deploying...
railway up

echo ✅ Railway deployment complete!
goto menu

:check_prerequisites
echo 🔍 Checking prerequisites...

where python >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed
    goto :eof
)

if not exist "requirements.txt" (
    echo ❌ requirements.txt not found
    goto :eof
)

if not exist "streamlit_app.py" (
    echo ❌ streamlit_app.py not found
    goto :eof
)

echo ✅ All prerequisites met!
exit /b 0

:exit
echo �� Goodbye!
exit /b 0 