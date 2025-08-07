#!/usr/bin/env python3
"""
Script to prepare the repository for GitHub
Shows which files will be included and excluded
"""

import os
import glob
from pathlib import Path

def check_git_files():
    """Check which files will be included in git"""
    
    print("🍅 Preparing repository for GitHub...")
    print("=" * 50)
    
    # Files that should be included
    essential_files = [
        'app.py',
        'run_web_app.py',
        'web_requirements.txt',
        'WEB_APP_README.md',
        'README.md',
        '.gitignore',
        'templates/web_app.html',
        'static/images/'
    ]
    
    # Files that should be excluded (large files)
    excluded_files = [
        'best_tomato_model.h5',
        'tomato_disease_model.h5',
        'train/',
        'valid/',
        'tomato_disease_env/',
        '__pycache__/',
        '*.pyc',
        '*.log'
    ]
    
    print("✅ Essential files to include:")
    for file in essential_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️  {file} (missing)")
    
    print("\n❌ Files that will be excluded (large/unnecessary):")
    for pattern in excluded_files:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            if os.path.exists(match):
                size = get_file_size(match)
                print(f"  - {match} ({size})")
    
    print("\n📁 Current directory structure:")
    print_directory_tree('.', max_depth=3)
    
    print("\n🚀 Ready for GitHub!")
    print("Run these commands to push to GitHub:")
    print("git init")
    print("git add .")
    print("git commit -m 'Initial commit: Tomato Disease Prediction Web App'")
    print("git branch -M main")
    print("git remote add origin https://github.com/yourusername/tomato-disease-prediction.git")
    print("git push -u origin main")

def get_file_size(path):
    """Get human readable file size"""
    try:
        if os.path.isfile(path):
            size = os.path.getsize(path)
        else:
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(path)
                      for filename in filenames)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return "Unknown size"

def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """Print directory tree structure"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            if item.startswith('.'):
                continue
                
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            
            if os.path.isdir(item_path):
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                new_prefix = prefix + ('    ' if is_last else '│   ')
                print_directory_tree(item_path, new_prefix, max_depth, current_depth + 1)
            else:
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
    except PermissionError:
        pass

if __name__ == "__main__":
    check_git_files() 