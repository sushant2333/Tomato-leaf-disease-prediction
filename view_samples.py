import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

def display_sample_images():
    """Display sample images from each class"""
    
    train_path = "train"
    classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, class_name in enumerate(classes[:12]):  # Show first 12 classes
        class_path = os.path.join(train_path, class_name)
        
        # Get first image from the class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        if image_files:
            # Load and display the first image
            img_path = image_files[0]
            try:
                img = mpimg.imread(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f'{class_name}\n({len(image_files)} images)', fontsize=10)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error loading\n{class_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No images\n{class_name}', 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(classes), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sample images saved as 'sample_images.png'")

def get_dataset_info():
    """Get detailed information about the dataset"""
    
    train_path = "train"
    valid_path = "valid"
    
    print("=" * 80)
    print("DETAILED TOMATO DISEASE DATASET INFORMATION")
    print("=" * 80)
    
    # Class descriptions
    class_descriptions = {
        'healthy': 'Healthy tomato leaves without any disease symptoms',
        'Bacterial_spot': 'Bacterial spot disease caused by Xanthomonas spp.',
        'Early_blight': 'Early blight caused by Alternaria solani',
        'Late_blight': 'Late blight caused by Phytophthora infestans',
        'Leaf_Mold': 'Leaf mold caused by Passalora fulva',
        'Septoria_leaf_spot': 'Septoria leaf spot caused by Septoria lycopersici',
        'Spider_mites Two-spotted_spider_mite': 'Spider mite damage (Two-spotted spider mite)',
        'Target_Spot': 'Target spot caused by Corynespora cassiicola',
        'Tomato_mosaic_virus': 'Tomato mosaic virus infection',
        'Tomato_Yellow_Leaf_Curl_Virus': 'Tomato yellow leaf curl virus infection',
        'powdery_mildew': 'Powdery mildew caused by Oidium lycopersicum'
    }
    
    print("\n📋 CLASS DESCRIPTIONS:")
    print("-" * 50)
    for class_name, description in class_descriptions.items():
        print(f"{class_name:35} : {description}")
    
    print("\n📊 DATASET STATISTICS:")
    print("-" * 50)
    
    # Count images in each class
    train_counts = {}
    valid_counts = {}
    
    for class_name in class_descriptions.keys():
        # Training data
        train_class_path = os.path.join(train_path, class_name)
        if os.path.exists(train_class_path):
            train_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
                train_images.extend(glob.glob(os.path.join(train_class_path, ext)))
            train_counts[class_name] = len(train_images)
        
        # Validation data
        valid_class_path = os.path.join(valid_path, class_name)
        if os.path.exists(valid_class_path):
            valid_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
                valid_images.extend(glob.glob(os.path.join(valid_class_path, ext)))
            valid_counts[class_name] = len(valid_images)
    
    # Print statistics
    total_train = sum(train_counts.values())
    total_valid = sum(valid_counts.values())
    
    print(f"Total classes: {len(class_descriptions)}")
    print(f"Total images: {total_train + total_valid:,}")
    print(f"Training images: {total_train:,}")
    print(f"Validation images: {total_valid:,}")
    print(f"Average images per class: {(total_train + total_valid) / len(class_descriptions):.0f}")
    
    print("\n📈 CLASS DISTRIBUTION:")
    print("-" * 50)
    print(f"{'Class':<35} {'Train':<8} {'Valid':<8} {'Total':<8} {'% of Total':<12}")
    print("-" * 80)
    
    for class_name in sorted(class_descriptions.keys()):
        train_count = train_counts.get(class_name, 0)
        valid_count = valid_counts.get(class_name, 0)
        total_count = train_count + valid_count
        percentage = (total_count / (total_train + total_valid)) * 100
        
        print(f"{class_name:<35} {train_count:<8} {valid_count:<8} {total_count:<8} {percentage:<12.1f}%")

if __name__ == "__main__":
    get_dataset_info()
    print("\n" + "=" * 80)
    print("Displaying sample images...")
    display_sample_images() 