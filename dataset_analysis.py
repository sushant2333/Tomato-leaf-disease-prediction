import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset():
    """Analyze the tomato disease prediction dataset structure"""
    
    # Define paths
    train_path = "train"
    valid_path = "valid"
    
    # Get all class directories
    train_classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    valid_classes = [d for d in os.listdir(valid_path) if os.path.isdir(os.path.join(valid_path, d))]
    
    print("=" * 60)
    print("TOMATO DISEASE PREDICTION DATASET ANALYSIS")
    print("=" * 60)
    
    # Analyze training data
    print("\n📊 TRAINING DATA ANALYSIS:")
    print("-" * 40)
    
    train_stats = {}
    total_train_images = 0
    
    for class_name in sorted(train_classes):
        class_path = os.path.join(train_path, class_name)
        image_files = []
        
        # Get all image files (jpg, jpeg, png, JPG)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        train_stats[class_name] = len(image_files)
        total_train_images += len(image_files)
        
        print(f"{class_name:30} : {len(image_files):5} images")
    
    print(f"\nTotal training images: {total_train_images}")
    
    # Analyze validation data
    print("\n📊 VALIDATION DATA ANALYSIS:")
    print("-" * 40)
    
    valid_stats = {}
    total_valid_images = 0
    
    for class_name in sorted(valid_classes):
        class_path = os.path.join(valid_path, class_name)
        image_files = []
        
        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        valid_stats[class_name] = len(image_files)
        total_valid_images += len(image_files)
        
        print(f"{class_name:30} : {len(image_files):5} images")
    
    print(f"\nTotal validation images: {total_valid_images}")
    
    # Dataset summary
    print("\n📋 DATASET SUMMARY:")
    print("-" * 40)
    print(f"Number of classes: {len(train_classes)}")
    print(f"Total images: {total_train_images + total_valid_images}")
    print(f"Training images: {total_train_images}")
    print(f"Validation images: {total_valid_images}")
    print(f"Train/Validation split: {total_train_images/(total_train_images + total_valid_images)*100:.1f}% / {total_valid_images/(total_train_images + total_valid_images)*100:.1f}%")
    
    # Class distribution analysis
    print("\n📈 CLASS DISTRIBUTION:")
    print("-" * 40)
    
    for class_name in sorted(train_stats.keys()):
        train_count = train_stats[class_name]
        valid_count = valid_stats.get(class_name, 0)
        total_count = train_count + valid_count
        train_pct = (train_count / total_count) * 100 if total_count > 0 else 0
        valid_pct = (valid_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"{class_name:30} : {total_count:5} total ({train_count:4} train, {valid_count:4} valid)")
        print(f"{'':30} : {train_pct:5.1f}% train, {valid_pct:5.1f}% valid")
    
    # File format analysis
    print("\n📁 FILE FORMAT ANALYSIS:")
    print("-" * 40)
    
    format_stats = defaultdict(int)
    
    for class_name in train_classes:
        class_path = os.path.join(train_path, class_name)
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            files = glob.glob(os.path.join(class_path, ext))
            format_stats[ext[1:]] += len(files)
    
    for format_name, count in sorted(format_stats.items()):
        percentage = (count / total_train_images) * 100
        print(f"{format_name:10} : {count:5} files ({percentage:5.1f}%)")
    
    # Sample image analysis
    print("\n🔍 SAMPLE IMAGE ANALYSIS:")
    print("-" * 40)
    
    # Check a few sample images from different classes
    sample_classes = ['healthy', 'Bacterial_spot', 'Late_blight', 'powdery_mildew']
    
    for class_name in sample_classes:
        class_path = os.path.join(train_path, class_name)
        if os.path.exists(class_path):
            # Get first few images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
                image_files.extend(glob.glob(os.path.join(class_path, ext)))
            
            if image_files:
                sample_files = image_files[:3]
                print(f"\n{class_name}:")
                for i, file_path in enumerate(sample_files, 1):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  Sample {i}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
    
    return train_stats, valid_stats

def create_visualization(train_stats, valid_stats):
    """Create visualizations for the dataset analysis"""
    
    # Prepare data for plotting
    classes = sorted(train_stats.keys())
    train_counts = [train_stats[cls] for cls in classes]
    valid_counts = [valid_stats.get(cls, 0) for cls in classes]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot for class distribution
    x = range(len(classes))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], train_counts, width, label='Training', alpha=0.8)
    ax1.bar([i + width/2 for i in x], valid_counts, width, label='Validation', alpha=0.8)
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Dataset Distribution by Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pie chart for overall distribution
    total_train = sum(train_counts)
    total_valid = sum(valid_counts)
    
    sizes = [total_train, total_valid]
    labels = ['Training', 'Validation']
    colors = ['#ff9999', '#66b3ff']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Overall Train/Validation Split')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Visualization saved as 'dataset_analysis.png'")

if __name__ == "__main__":
    train_stats, valid_stats = analyze_dataset()
    create_visualization(train_stats, valid_stats) 