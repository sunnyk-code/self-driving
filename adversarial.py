import numpy as np
import os
import tensorflow as tf
from scipy.ndimage import gaussian_filter

def generate_adversarial_fgsm(image, model, epsilon=0.01, targeted=False):
    """
    Generate an adversarial example using a numerical approximation of FGSM.
    Since we're working with a frozen TensorFlow graph, we use a simplified approach
    that approximates the gradient with respect to the input.
    
    Args:
        image: Input image as numpy array (uint8)
        model: Model with a predict method
        epsilon: Attack strength parameter (between 0 and 1)
        targeted: Whether to perform a targeted attack (not fully implemented)
        
    Returns:
        Adversarial example as numpy array (uint8), original prediction
    """
    print(f"Generating adversarial example with epsilon={epsilon}")
    
    # Convert to float32 for manipulation
    image_float = image.astype(np.float32)
    
    # Get original prediction
    orig_pred = model.predict(image)
    print(f"Original prediction shape: {orig_pred.shape}")
    
    # Approximate the gradient using a simple numerical approach
    # We'll compute how small changes in each pixel affect the prediction
    perturbation = np.zeros_like(image_float)
    
    # For efficiency, we'll use a random sampling approach to estimate the gradient
    # Use a small number of random pixels to approximate the gradient direction
    h, w, c = image.shape
    num_samples = min(1000, h*w//100)  # Sample 1% of pixels or 1000, whichever is smaller
    
    print(f"Computing numerical gradient approximation using {num_samples} samples...")
    
    # Randomly select pixels to compute gradient estimates for
    sample_y = np.random.randint(0, h, num_samples)
    sample_x = np.random.randint(0, w, num_samples)
    
    # Small delta for numerical differentiation
    delta = 1.0
    
    # For each sampled pixel, compute the gradient direction
    for i in range(num_samples):
        y, x = sample_y[i], sample_x[i]
        
        # For each color channel
        for ch in range(c):
            # Create a copy with a small perturbation in this pixel
            perturbed = image_float.copy()
            perturbed[y, x, ch] += delta
            
            # Get prediction on perturbed image
            new_pred = model.predict(perturbed.astype(np.uint8))
            
            # Calculate how much the prediction changed (simple difference)
            # For a proper loss, we would use cross-entropy, but this is simpler and works
            pred_diff = np.sum(new_pred != orig_pred)
            
            # If targeted, we want to minimize the difference (move towards target)
            # If untargeted, we want to maximize the difference (move away from original)
            gradient_direction = -1 if targeted else 1
            
            # Set the gradient direction for this pixel
            if pred_diff > 0:
                perturbation[y, x, ch] = gradient_direction * delta
    
    # Propagate the gradient estimates to nearby pixels (smoothing)
    perturbation = gaussian_filter(perturbation, sigma=3.0)
    
    # Apply FGSM step: add epsilon * sign(gradient) to the image
    perturbed_image = image_float + epsilon * 255.0 * np.sign(perturbation)
    
    # Ensure valid image by clipping to [0, 255]
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    
    # For analysis, calculate the L-infinity norm of the perturbation
    applied_perturbation = perturbed_image - image
    perturbation_linf = np.max(np.abs(applied_perturbation))
    print(f"Perturbation L-infinity norm: {perturbation_linf:.2f}")
    
    return perturbed_image, orig_pred

def calculate_prediction_difference(pred1, pred2):
    """Calculate the percentage of pixels that differ between two predictions."""
    if pred1.shape != pred2.shape:
        raise ValueError(f"Predictions have different shapes: {pred1.shape} vs {pred2.shape}")
    
    num_diff_pixels = np.sum(pred1 != pred2)
    total_pixels = np.prod(pred1.shape)
    diff_percentage = (num_diff_pixels / total_pixels) * 100
    
    return diff_percentage 