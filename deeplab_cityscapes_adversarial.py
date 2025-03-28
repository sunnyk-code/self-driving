#!/usr/bin/env python3
"""
Script to test adversarial examples using DeepLabV3Plus-Pytorch models

TO USE THIS CLONE THIS REPO: https://github.com/VainF/DeepLabV3Plus-Pytorch?tab=readme-ov-file
AND DOWNLOAD THIS MODEL AND PUT IN CHECKPOINTS FOLDER: https://www.dropbox.com/scl/fi/jo4nhw3h6lcg8t2ckarae/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?e=2&rlkey=7qnzapkshyofrgfa1ls7vot6j

"""
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Import our modules
from deeplab_cityscapes_pretrained import DeepLabV3PlusModel
from visualization import visualize_results, visualize_perturbation_only
from cityscapes_utils import create_colormap, CITYSCAPES_COLORS

# Constants
RESULTS_DIR = 'deeplabv3plus_adversarial_results'
EPSILON_VALUES = [0.01, 0.1, 0.3]
CHECKPOINT_PATH = './checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
# If you're using the ResNet101 version, use this checkpoint instead:
# CHECKPOINT_PATH = './DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth'

def save_image(img_array, file_path):
    """Save a numpy array as an image."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    Image.fromarray(img_array.astype(np.uint8)).save(file_path)

def calculate_prediction_difference(pred1, pred2):
    """Calculate the percentage of pixels that differ between two predictions."""
    if pred1.shape != pred2.shape:
        raise ValueError(f"Predictions must have the same shape: {pred1.shape} vs {pred2.shape}")
    
    diff_pixels = np.sum(pred1 != pred2)
    total_pixels = pred1.size
    percentage = (diff_pixels / total_pixels) * 100
    return percentage

def test_adversarial_examples():
    # Create DeepLabV3+ model
    print("Loading DeepLabV3+ model...")
    model = DeepLabV3PlusModel(
        model_name='deeplabv3plus_mobilenet',  # Change to deeplabv3plus_resnet101 for the ResNet101 version
        checkpoint_path=CHECKPOINT_PATH
    )
    
    # Create output directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Sample images
    img_dir = '../leftImg8bit_trainvaltest/leftImg8bit/test'
    sample_images = []
    
    # Find a few sample images
    for city in os.listdir(img_dir):
        city_path = os.path.join(img_dir, city)
        if os.path.isdir(city_path):
            img_files = [os.path.join(city_path, f) for f in os.listdir(city_path) 
                       if f.endswith('leftImg8bit.png')]
            sample_images.extend(img_files[:2])  # Take 2 images from each city
            if len(sample_images) >= 3:  # Limit to a few sample images
                break
    
    print(f"Testing on {len(sample_images)} sample images")
    
    # Process each sample image
    for img_path in sample_images:
        print(f"\nProcessing {img_path}")
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load image
        img = np.array(Image.open(img_path))
        
        # Run prediction on original image
        print("Running prediction on original image...")
        orig_pred = model.predict(img)
        
        # Save original prediction
        orig_img_path = os.path.join(RESULTS_DIR, f'{base_filename}_original.png')
        orig_seg_path = os.path.join(RESULTS_DIR, f'{base_filename}_original_seg.png')
        save_image(img, orig_img_path)
        
        # Try to get colormap for segmentation visualization
        try:
            # Use the directly imported CITYSCAPES_COLORS constant
            colored_seg = create_colormap(orig_pred, CITYSCAPES_COLORS)
            save_image(colored_seg, orig_seg_path)
        except Exception as e:
            print(f"Warning: Could not create colored segmentation: {e}")
            # Save grayscale if colormap not available
            save_image((orig_pred * 10).astype(np.uint8), orig_seg_path)
        
        # Generate adversarial examples with different epsilons
        for epsilon in EPSILON_VALUES:
            print(f"Generating adversarial example with epsilon={epsilon}")
            
            try:
                # Generate adversarial example
                adv_img, perturbation = model.generate_adversarial_fgsm(img, epsilon=epsilon)
                
                # Predict on adversarial example
                adv_pred = model.predict(adv_img)
                
                # Create output directory for this epsilon value
                eps_dir = os.path.join(RESULTS_DIR, f'epsilon_{epsilon}')
                os.makedirs(eps_dir, exist_ok=True)
                
                # Save adversarial image and prediction
                adv_img_path = os.path.join(eps_dir, f'{base_filename}_adversarial.png')
                adv_seg_path = os.path.join(eps_dir, f'{base_filename}_adversarial_seg.png')
                save_image(adv_img, adv_img_path)
                
                try:
                    colored_adv_seg = create_colormap(adv_pred, CITYSCAPES_COLORS)
                    save_image(colored_adv_seg, adv_seg_path)
                except:
                    save_image((adv_pred * 10).astype(np.uint8), adv_seg_path)
                
                # Save perturbation visualization
                pert_vis_path = os.path.join(eps_dir, f'{base_filename}_perturbation.png')
                pert_vis = visualize_perturbation_only(perturbation)
                save_image(pert_vis, pert_vis_path)
                
                # Calculate difference
                diff_percentage = calculate_prediction_difference(orig_pred, adv_pred)
                print(f"Epsilon={epsilon}: {diff_percentage:.2f}% of pixels changed in prediction")
                
                #Save comparison
                comparison_path = os.path.join(eps_dir, f'{base_filename}_comparison.png')
                try:
                    
                    # Create colormap - normalize colors to [0,1] range for matplotlib
                    cityscapes_cmap = ListedColormap(np.array(CITYSCAPES_COLORS) / 255.0)
                    
                    # Call visualize_results with the proper parameters
                    visualize_results(
                        original_image=img, 
                        original_pred=orig_pred,
                        adversarial_image=adv_img,
                        adversarial_pred=adv_pred,
                        colormap=cityscapes_cmap,
                        title=f"Adversarial Attack (ε={epsilon})",
                        save_path=comparison_path
                    )
                    
                except Exception as e:
                    print(f"Error in visualization: {e}")
                    # Create simple side-by-side comparison as fallback
                    plt.figure(figsize=(12, 6))
                    plt.subplot(2, 2, 1)
                    plt.imshow(img)
                    plt.title("Original Image")
                    plt.subplot(2, 2, 2)
                    plt.imshow(adv_img)
                    plt.title(f"Adversarial Image (ε={epsilon})")
                    plt.subplot(2, 2, 3)
                    # Use 'tab20' colormap instead for the fallback
                    plt.imshow(orig_pred, cmap='tab20')
                    plt.title("Original Prediction")
                    plt.subplot(2, 2, 4)
                    plt.imshow(adv_pred, cmap='tab20')
                    plt.title(f"Adversarial Prediction")
                    plt.tight_layout()
                    plt.savefig(comparison_path)
                    plt.close()
            
            except Exception as e:
                print(f"Error with epsilon={epsilon}: {e}")
    
    # Cleanup
    model.close()
    print("Testing complete!")

if __name__ == "__main__":
    test_adversarial_examples()