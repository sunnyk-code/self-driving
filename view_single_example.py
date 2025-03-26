#!/usr/bin/env python3
"""
Script to visualize a single example with FGSM adversarial perturbation.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import our modules
from model_utils import DeepLabModel
from adversarial import generate_adversarial_fgsm
from cityscapes_utils import create_colormap, CITYSCAPES_COLORS, load_cityscapes_image

def view_single_example(image_path, epsilon=0.01, output_dir='single_example_results'):
    """
    Process a single image and visualize the effect of FGSM adversarial perturbation.
    
    Args:
        image_path: Path to input image
        epsilon: Perturbation strength (default: 0.01)
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model_path = os.path.join('models', 'deeplabv3_cityscapes_train', 'frozen_inference_graph.pb')
    print(f"Loading model from {model_path}")
    model = DeepLabModel(model_path)
    
    # Load image
    print(f"Loading image from {image_path}")
    original_image = load_cityscapes_image(image_path)
    
    # Get original prediction
    print("Running original prediction...")
    original_pred = model.predict(original_image)
    
    # Generate adversarial examples with different epsilon values
    epsilons = [0.005, 0.01, 0.02, 0.05] if epsilon is None else [epsilon]
    adv_images = []
    adv_preds = []
    
    for eps in epsilons:
        print(f"Generating adversarial example with epsilon={eps}")
        adv_image, _ = generate_adversarial_fgsm(original_image, model, eps)
        adv_pred = model.predict(adv_image)
        
        adv_images.append(adv_image)
        adv_preds.append(adv_pred)
        
        # Calculate difference
        diff_percentage = np.sum(original_pred != adv_pred) / np.prod(original_pred.shape) * 100
        print(f"Prediction difference: {diff_percentage:.2f}% of pixels changed")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Convert segmentation predictions to color
    original_pred_color = create_colormap(original_pred, CITYSCAPES_COLORS)
    adv_preds_color = [create_colormap(pred, CITYSCAPES_COLORS) for pred in adv_preds]
    
    # 1. Visualize original image and prediction
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(original_pred_color)
    plt.title("Original Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original.png"))
    plt.close()
    
    # 2. Visualize perturbations and their effect for each epsilon
    for i, eps in enumerate(epsilons):
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Adversarial image
        plt.subplot(1, 3, 2)
        plt.imshow(adv_images[i])
        plt.title(f"Adversarial Image (ε={eps})")
        plt.axis('off')
        
        # Difference between original and adversarial
        plt.subplot(1, 3, 3)
        # Amplify the difference for visibility
        diff = np.abs(original_image.astype(np.float32) - adv_images[i].astype(np.float32))
        # Scale to full range for visibility
        diff = diff * 10  # Amplify
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        plt.imshow(diff)
        plt.title(f"Perturbation (amplified 10x)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_image_eps_{eps}.png"))
        plt.close()
        
        # 3. Visualize segmentation changes
        plt.figure(figsize=(15, 5))
        
        # Original prediction
        plt.subplot(1, 3, 1)
        plt.imshow(original_pred_color)
        plt.title("Original Prediction")
        plt.axis('off')
        
        # Adversarial prediction
        plt.subplot(1, 3, 2)
        plt.imshow(adv_preds_color[i])
        plt.title(f"Adversarial Prediction (ε={eps})")
        plt.axis('off')
        
        # Highlight differences
        plt.subplot(1, 3, 3)
        mask = original_pred != adv_preds[i]
        diff_highlight = np.zeros_like(original_image)
        diff_highlight[..., 0] = 255 * mask  # Red channel for differences
        plt.imshow(diff_highlight)
        plt.title(f"Changed Pixels (red)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_pred_eps_{eps}.png"))
        plt.close()
    
    # 4. Create a grid comparing all epsilon values
    if len(epsilons) > 1:
        rows = len(epsilons)
        plt.figure(figsize=(15, 5 * rows))
        
        for i, eps in enumerate(epsilons):
            # Adversarial image
            plt.subplot(rows, 3, i*3 + 1)
            plt.imshow(adv_images[i])
            plt.title(f"Adversarial (ε={eps})")
            plt.axis('off')
            
            # Adversarial prediction
            plt.subplot(rows, 3, i*3 + 2)
            plt.imshow(adv_preds_color[i])
            plt.title(f"Prediction (ε={eps})")
            plt.axis('off')
            
            # Changed pixels
            plt.subplot(rows, 3, i*3 + 3)
            mask = original_pred != adv_preds[i]
            diff_percentage = np.sum(mask) / np.prod(mask.shape) * 100
            diff_highlight = np.zeros_like(original_image)
            diff_highlight[..., 0] = 255 * mask  # Red channel
            plt.imshow(diff_highlight)
            plt.title(f"Changed: {diff_percentage:.2f}%")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epsilon_comparison.png"))
        plt.close()
    
    # Close the model
    model.close()
    
    print(f"Results saved to {output_dir}")
    print(f"To view the results, open the PNG files in the {output_dir} directory")

def main():
    parser = argparse.ArgumentParser(description="Visualize FGSM adversarial examples")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--epsilon", type=float, default=None,
                       help="Perturbation strength (default: multiple values)")
    parser.add_argument("--output_dir", type=str, default="single_example_results",
                       help="Directory to save results")
    args = parser.parse_args()
    
    view_single_example(args.image_path, args.epsilon, args.output_dir)

if __name__ == "__main__":
    main() 