#!/usr/bin/env python3
"""
Numerical approximation of Fast Gradient Sign Method (FGSM) for semantic segmentation.
This implementation calculates gradients by evaluating the model on small perturbations.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import time
from tqdm import tqdm

def calculate_loss(predictions, targets):
    """
    Calculate cross-entropy loss between predictions and targets.
    
    Args:
        predictions: Model predictions
        targets: Target labels
        
    Returns:
        Loss value
    """
    # Simple cross-entropy proxy: sum of absolute differences
    return np.sum(np.abs(predictions - targets))

def generate_numerical_fgsm(image, model, epsilon=0.01, targeted=False, num_samples=1000, step_size=1.0):
    """
    Generate an adversarial example using a numerical approximation of FGSM.
    
    Args:
        image: Input image as numpy array (uint8)
        model: Model with predict method
        epsilon: Perturbation strength (0 to 1)
        targeted: Whether to use targeted attack (default: False)
        num_samples: Number of pixel samples for gradient estimation
        step_size: Size of perturbation for gradient estimation
        
    Returns:
        Adversarial example as numpy array (uint8), original prediction
    """
    print(f"Generating numerical FGSM with epsilon={epsilon}, samples={num_samples}")
    start_time = time.time()
    
    # Convert to float32 for manipulation
    image_float = image.astype(np.float32)
    h, w, c = image.shape
    
    # Get original prediction
    orig_pred = model.predict(image)
    
    # Initialize gradient estimate
    grad_estimate = np.zeros_like(image_float)
    
    # For targeted attack, create a target different from the original
    if targeted:
        # Shift all class labels by 1 mod num_classes
        print("Using targeted attack (inverting gradient direction)")
        target_pred = (orig_pred + 1) % 19  # Assuming 19 classes for Cityscapes
    else:
        # Use original prediction as target for loss calculation
        target_pred = orig_pred
    
    # Calculate baseline loss
    baseline_loss = calculate_loss(orig_pred, target_pred)
    
    # Randomly sample pixels for gradient estimation
    print("Estimating gradients...")
    y_indices = np.random.randint(0, h, num_samples)
    x_indices = np.random.randint(0, w, num_samples)
    
    for i in tqdm(range(num_samples)):
        y, x = y_indices[i], x_indices[i]
        
        # Estimate gradient for each color channel
        for ch in range(c):
            # Create a copy with positive perturbation
            perturbed_pos = image_float.copy()
            perturbed_pos[y, x, ch] += step_size
            perturbed_pos = np.clip(perturbed_pos, 0, 255)
            
            # Get prediction on perturbed image
            perturbed_pred = model.predict(perturbed_pos.astype(np.uint8))
            
            # Calculate loss
            perturbed_loss = calculate_loss(perturbed_pred, target_pred)
            
            # Estimate gradient
            grad_estimate[y, x, ch] = perturbed_loss - baseline_loss
    
    # Smooth the gradient estimate
    from scipy.ndimage import gaussian_filter
    for ch in range(c):
        grad_estimate[:, :, ch] = gaussian_filter(grad_estimate[:, :, ch], sigma=3)
    
    # Propagate gradient estimates to nearby pixels
    print("Propagating gradients...")
    full_grad = np.zeros_like(image_float)
    
    # Use a simple distance-based propagation
    for i in tqdm(range(num_samples)):
        y, x = y_indices[i], x_indices[i]
        
        # Define a local region around sampled pixel
        y_min, y_max = max(0, y - 10), min(h, y + 10)
        x_min, x_max = max(0, x - 10), min(w, x + 10)
        
        # Apply gradient to local region with distance-based falloff
        for y_local in range(y_min, y_max):
            for x_local in range(x_min, x_max):
                # Calculate distance-based weight
                dist = np.sqrt((y - y_local)**2 + (x - x_local)**2)
                weight = max(0, 1 - dist / 10)  # Linear falloff
                
                # Apply weighted gradient
                full_grad[y_local, x_local] += weight * grad_estimate[y, x]
    
    # Create FGSM perturbation: epsilon * sign(gradient)
    sign_direction = -1 if targeted else 1
    perturbation = sign_direction * epsilon * 255.0 * np.sign(full_grad)
    
    # Apply perturbation
    perturbed_image = image_float + perturbation
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    
    # Calculate how long it took
    elapsed_time = time.time() - start_time
    print(f"Generation time: {elapsed_time:.2f} seconds")
    
    return perturbed_image, orig_pred

def run_numerical_fgsm_demo(image_path, model_path, epsilon=0.01, targeted=False, 
                          num_samples=1000, output_dir='numerical_fgsm_results'):
    """
    Run numerical FGSM demo on a single image.
    
    Args:
        image_path: Path to input image
        model_path: Path to model
        epsilon: Perturbation strength
        targeted: Whether to use targeted attack
        num_samples: Number of pixel samples for gradient estimation
        output_dir: Directory to save results
    """
    from model_utils import DeepLabModel
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = DeepLabModel(model_path)
    
    # Load the image
    print(f"Loading image from {image_path}")
    image = np.array(Image.open(image_path).convert('RGB'))
    
    try:
        # Generate adversarial example
        adv_image, orig_pred = generate_numerical_fgsm(
            image, model, epsilon, targeted, num_samples
        )
        
        # Get prediction on adversarial example
        adv_pred = model.predict(adv_image)
        
        # Calculate difference metrics
        diff_mask = orig_pred != adv_pred
        diff_percentage = np.sum(diff_mask) / np.prod(diff_mask.shape) * 100
        
        print(f"Segmentation changed for {diff_percentage:.2f}% of pixels")
        
        # Get name for saving
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Visualize results
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Adversarial image
        plt.subplot(2, 3, 2)
        plt.imshow(adv_image)
        plt.title(f"Adversarial Image (ε={epsilon})")
        plt.axis('off')
        
        # Perturbation (amplified for visibility)
        plt.subplot(2, 3, 3)
        perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
        perturbation_amp = np.clip(perturbation * 10, 0, 255).astype(np.uint8)
        plt.imshow(perturbation_amp)
        plt.title("Perturbation (10x)")
        plt.axis('off')
        
        # Original prediction
        plt.subplot(2, 3, 4)
        plt.imshow(orig_pred, cmap='nipy_spectral')
        plt.title("Original Prediction")
        plt.axis('off')
        
        # Adversarial prediction
        plt.subplot(2, 3, 5)
        plt.imshow(adv_pred, cmap='nipy_spectral')
        plt.title("Adversarial Prediction")
        plt.axis('off')
        
        # Changed pixels visualization
        plt.subplot(2, 3, 6)
        changed_vis = np.zeros_like(image)
        changed_vis[..., 0] = 255 * diff_mask  # Red channel
        plt.imshow(changed_vis)
        plt.title(f"Changed Pixels: {diff_percentage:.2f}%")
        plt.axis('off')
        
        plt.suptitle(f"Numerical FGSM Results (ε={epsilon}, {'Targeted' if targeted else 'Untargeted'})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{image_name}_fgsm_eps_{epsilon}.png")
        plt.savefig(output_path)
        plt.close()
        
        # Save adversarial image
        adv_path = os.path.join(output_dir, f"{image_name}_adversarial_eps_{epsilon}.png")
        Image.fromarray(adv_image).save(adv_path)
        
        # Save summary file
        summary_path = os.path.join(output_dir, f"{image_name}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Numerical FGSM Results Summary\n")
            f.write(f"=============================\n\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Epsilon: {epsilon}\n")
            f.write(f"Attack type: {'Targeted' if targeted else 'Untargeted'}\n")
            f.write(f"Number of samples: {num_samples}\n\n")
            f.write(f"Results:\n")
            f.write(f"  - Pixels changed: {diff_percentage:.2f}%\n")
            f.write(f"  - Max perturbation: {np.max(perturbation):.2f}\n")
            f.write(f"  - Mean perturbation: {np.mean(perturbation):.2f}\n")
        
        print(f"Results saved to {output_dir}")
        print(f"Visualization: {output_path}")
        print(f"Adversarial image: {adv_path}")
        print(f"Summary: {summary_path}")
        
    finally:
        # Clean up
        model.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Numerical FGSM for semantic segmentation")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--model_path", type=str, 
                       default="models/deeplabv3_cityscapes_train/frozen_inference_graph.pb",
                       help="Path to model")
    parser.add_argument("--epsilon", type=float, default=0.01,
                       help="Perturbation strength (default: 0.01)")
    parser.add_argument("--targeted", action="store_true",
                       help="Use targeted attack (default: False)")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of pixel samples for gradient estimation (default: 1000)")
    parser.add_argument("--output_dir", type=str, default="numerical_fgsm_results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    run_numerical_fgsm_demo(
        args.image_path,
        args.model_path,
        args.epsilon,
        args.targeted,
        args.num_samples,
        args.output_dir
    ) 