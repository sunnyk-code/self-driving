#!/usr/bin/env python3
"""
True Fast Gradient Sign Method (FGSM) implementation for semantic segmentation.
This implementation computes actual gradients with respect to the model loss.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import time

def generate_one_hot(prediction, num_classes):
    """
    Convert a class prediction to one-hot encoding.
    
    Args:
        prediction: Class prediction array of shape (H, W)
        num_classes: Number of classes
    
    Returns:
        One-hot encoded tensor of shape (H, W, num_classes)
    """
    # Create a one-hot encoded version of the prediction
    one_hot = np.zeros(prediction.shape + (num_classes,), dtype=np.float32)
    
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            class_idx = prediction[i, j]
            one_hot[i, j, class_idx] = 1.0
            
    return one_hot

def generate_true_fgsm(image, model, epsilon=0.01, targeted=False, num_classes=19):
    """
    Generate an adversarial example using the true Fast Gradient Sign Method (FGSM).
    
    Args:
        image: Input image as numpy array (uint8)
        model: DeepLabModel with gradient computation capability
        epsilon: Strength of the perturbation (0 to 1)
        targeted: Whether to perform a targeted attack (False for untargeted)
        num_classes: Number of classes in the segmentation model
        
    Returns:
        Adversarial example as numpy array (uint8), original prediction
    """
    print(f"Generating true FGSM adversarial example with epsilon={epsilon}")
    start_time = time.time()
    
    # Convert to float32 for manipulation
    image_float = image.astype(np.float32)
    
    # Get original prediction
    orig_pred = model.predict(image)
    print(f"Original prediction shape: {orig_pred.shape}")
    
    # Create target for gradient computation
    if targeted:
        # For targeted attack, use a modified target
        # Example: shift all classes by 1 (mod num_classes)
        target_pred = (orig_pred + 1) % num_classes
        print("Using targeted attack")
    else:
        # For untargeted attack, use the original prediction as target
        # The loss function will try to maximize the difference from this target
        target_pred = orig_pred
        print("Using untargeted attack")
    
    # Convert to one-hot encoding
    target_one_hot = generate_one_hot(target_pred, num_classes)
    print(f"Target one-hot shape: {target_one_hot.shape}")
    
    # Compute gradients of the loss with respect to the input image
    print("Computing gradients...")
    gradients = model.compute_gradients(image, target_one_hot)
    print(f"Gradients shape: {gradients.shape}")
    
    # For untargeted attack: maximize loss by adding gradient
    # For targeted attack: minimize loss by subtracting gradient
    sign_direction = -1 if targeted else 1
    
    # FGSM update: image += sign_direction * epsilon * sign(gradients)
    perturbation = sign_direction * epsilon * 255.0 * np.sign(gradients[0])
    perturbed_image = image_float + perturbation
    
    # Ensure valid image by clipping to [0, 255]
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    
    elapsed_time = time.time() - start_time
    print(f"FGSM generation time: {elapsed_time:.2f} seconds")
    
    return perturbed_image, orig_pred

def visualize_fgsm_results(image, adv_image, orig_pred, adv_pred, epsilon, output_dir='true_fgsm_results'):
    """
    Visualize results of FGSM attack.
    
    Args:
        image: Original image
        adv_image: Adversarial image
        orig_pred: Original prediction
        adv_pred: Adversarial prediction
        epsilon: Epsilon value used
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate difference between predictions
    diff_mask = orig_pred != adv_pred
    diff_percentage = np.sum(diff_mask) / np.prod(diff_mask.shape) * 100
    
    # Calculate perturbation
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    perturbation_norm = np.max(perturbation)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Original image and prediction
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(orig_pred, cmap='nipy_spectral')
    plt.title("Original Prediction")
    plt.axis('off')
    
    # Adversarial image and prediction
    plt.subplot(2, 3, 2)
    plt.imshow(adv_image)
    plt.title(f"Adversarial Image (ε={epsilon})")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(adv_pred, cmap='nipy_spectral')
    plt.title(f"Adversarial Prediction")
    plt.axis('off')
    
    # Perturbation and difference visualization
    plt.subplot(2, 3, 3)
    plt.imshow(perturbation * 10, cmap='hot')  # Amplified for visibility
    plt.title(f"Perturbation (10x)\nMax = {perturbation_norm:.1f}")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    diff_vis = np.zeros_like(image)
    diff_vis[..., 0] = 255 * diff_mask  # Red channel
    plt.imshow(diff_vis)
    plt.title(f"Changed Pixels: {diff_percentage:.2f}%")
    plt.axis('off')
    
    plt.suptitle(f"True FGSM Results (ε={epsilon})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save visualization
    output_path = os.path.join(output_dir, f"true_fgsm_epsilon_{epsilon}.png")
    plt.savefig(output_path)
    plt.close()
    
    # Save individual images
    Image.fromarray(adv_image).save(os.path.join(output_dir, f"adversarial_eps_{epsilon}.png"))
    
    return diff_percentage, perturbation_norm

def run_true_fgsm_demo(image_path, model, epsilon=0.01, targeted=False, output_dir='true_fgsm_results'):
    """
    Run FGSM demo on a single image.
    
    Args:
        image_path: Path to input image
        model: DeepLabModel instance
        epsilon: Perturbation strength
        targeted: Whether to perform targeted attack
        output_dir: Directory to save results
    
    Returns:
        Difference percentage, perturbation norm
    """
    # Load image
    print(f"Loading image: {image_path}")
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Generate adversarial example
    adv_image, orig_pred = generate_true_fgsm(
        image, model, epsilon, targeted, num_classes=19
    )
    
    # Get prediction on adversarial example
    adv_pred = model.predict(adv_image)
    
    # Visualize results
    diff_percentage, perturbation_norm = visualize_fgsm_results(
        image, adv_image, orig_pred, adv_pred, epsilon, output_dir
    )
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"True FGSM Results Summary\n")
        f.write(f"=======================\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Attack type: {'Targeted' if targeted else 'Untargeted'}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Pixels changed: {diff_percentage:.2f}%\n")
        f.write(f"  - Perturbation max value: {perturbation_norm:.2f}\n")
    
    print(f"Results saved to {output_dir}")
    return diff_percentage, perturbation_norm

if __name__ == "__main__":
    import argparse
    from model_utils import DeepLabModel
    
    parser = argparse.ArgumentParser(description="True FGSM for semantic segmentation")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Perturbation strength (default: 0.01)")
    parser.add_argument("--targeted", action="store_true",
                        help="Use targeted attack")
    parser.add_argument("--output_dir", type=str, default="true_fgsm_results",
                        help="Directory to save results")
    parser.add_argument("--model_path", type=str, 
                        default="models/deeplabv3_cityscapes_train/frozen_inference_graph.pb",
                        help="Path to DeepLabV3 model")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = DeepLabModel(args.model_path)
    
    try:
        # Run demo
        run_true_fgsm_demo(
            args.image_path, model, args.epsilon, args.targeted, args.output_dir
        )
    finally:
        # Clean up
        model.close() 