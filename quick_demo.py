#!/usr/bin/env python3
"""
Quick demo for visualizing effects of adversarial perturbations on semantic segmentation.
This script uses the fast FGSM implementation for quick demonstration purposes.
"""
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import our modules
from model_utils import DeepLabModel
from fast_fgsm import generate_fast_fgsm
from cityscapes_utils import create_colormap, CITYSCAPES_COLORS

def run_quick_demo(image_path, epsilon=0.02, method='structured', output_dir='quick_demo_results'):
    """
    Run a quick demonstration of adversarial perturbations on semantic segmentation.
    
    Args:
        image_path: Path to input image
        epsilon: Perturbation strength (default: 0.02)
        method: Type of perturbation pattern ('structured', 'edge', 'frequency')
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("FGSM vs Semantic Segmentation - Quick Demo")
    print("==========================================")
    
    # Step 1: Load model
    print("\nStep 1: Loading semantic segmentation model...")
    model_path = os.path.join('models', 'deeplabv3_cityscapes_train', 'frozen_inference_graph.pb')
    model = DeepLabModel(model_path)
    
    # Step 2: Load and prepare image
    print(f"\nStep 2: Loading image: {image_path}")
    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Image shape: {image.shape}")
    
    # Step 3: Generate adversarial example
    print(f"\nStep 3: Generating adversarial example using {method} pattern with epsilon={epsilon}...")
    start_time = time.time()
    adv_image = generate_fast_fgsm(image, epsilon, method)
    gen_time = time.time() - start_time
    print(f"Generation time: {gen_time:.2f} seconds")
    
    # Save the original and adversarial images
    Image.fromarray(image).save(os.path.join(output_dir, "original.png"))
    Image.fromarray(adv_image).save(os.path.join(output_dir, f"adversarial_{method}_eps_{epsilon}.png"))
    
    # Step 4: Run predictions
    print("\nStep 4: Running predictions on original and adversarial images...")
    start_time = time.time()
    orig_pred = model.predict(image)
    adv_pred = model.predict(adv_image)
    pred_time = time.time() - start_time
    print(f"Prediction time: {pred_time:.2f} seconds")
    
    # Convert predictions to color
    orig_pred_color = create_colormap(orig_pred, CITYSCAPES_COLORS)
    adv_pred_color = create_colormap(adv_pred, CITYSCAPES_COLORS)
    
    # Step 5: Analyze and visualize
    print("\nStep 5: Analyzing and visualizing results...")
    
    # Calculate difference metrics
    pixel_diff = np.sum(orig_pred != adv_pred)
    total_pixels = np.prod(orig_pred.shape)
    diff_percentage = (pixel_diff / total_pixels) * 100
    
    print(f"Segmentation changes:")
    print(f"  - Number of pixels changed: {pixel_diff} out of {total_pixels}")
    print(f"  - Percentage of pixels changed: {diff_percentage:.2f}%")
    
    # Create a mask of changed pixels for visualization
    diff_mask = orig_pred != adv_pred
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Row 1: Images
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(adv_image)
    plt.title(f"Adversarial Image\n(ε={epsilon}, {method})")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    # Visualize the perturbation (amplified for visibility)
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    perturbation = perturbation * 10  # Amplify
    perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)
    plt.imshow(perturbation)
    plt.title("Perturbation (10x)")
    plt.axis('off')
    
    # Row 2: Predictions
    plt.subplot(2, 3, 4)
    plt.imshow(orig_pred_color)
    plt.title("Original Prediction")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(adv_pred_color)
    plt.title("Adversarial Prediction")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    # Visualize changed pixels
    diff_highlight = np.zeros_like(image)
    diff_highlight[..., 0] = 255 * diff_mask  # Red channel for differences
    plt.imshow(diff_highlight)
    plt.title(f"Changed Pixels: {diff_percentage:.2f}%")
    plt.axis('off')
    
    plt.suptitle(f"FGSM vs Semantic Segmentation (ε={epsilon}, {method})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save visualization
    result_path = os.path.join(output_dir, f"results_{method}_eps_{epsilon}.png")
    plt.savefig(result_path)
    plt.close()
    
    print(f"\nResults saved to {output_dir}")
    print(f"Main visualization: {result_path}")
    
    # Save additional information
    with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
        f.write("FGSM vs Semantic Segmentation - Results Summary\n")
        f.write("=============================================\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Image dimensions: {image.shape}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Epsilon: {epsilon}\n\n")
        
        f.write("Timing:\n")
        f.write(f"  - Adversarial example generation: {gen_time:.2f} seconds\n")
        f.write(f"  - Model prediction: {pred_time:.2f} seconds\n\n")
        
        f.write("Effectiveness:\n")
        f.write(f"  - Pixels changed: {pixel_diff} out of {total_pixels}\n")
        f.write(f"  - Percentage changed: {diff_percentage:.2f}%\n")
    
    # Clean up
    model.close()

def main():
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description="Quick demo of FGSM against semantic segmentation")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="Perturbation strength (default: 0.02)")
    parser.add_argument("--method", type=str, default="structured",
                        choices=["structured", "edge", "frequency"],
                        help="Type of perturbation pattern (default: structured)")
    parser.add_argument("--output_dir", type=str, default="quick_demo_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    run_quick_demo(args.image_path, args.epsilon, args.method, args.output_dir)

if __name__ == "__main__":
    main() 