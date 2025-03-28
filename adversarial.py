import numpy as np
import tensorflow as tf
from PIL import Image

def generate_adversarial_fgsm(image, model, epsilon=0.01, targeted=False):
    """
    Generate an adversarial example using the true Fast Gradient Sign Method (FGSM)
    by utilizing model gradients.
    
    Args:
        image: Input image as numpy array
        model: DeepLabModel instance
        epsilon: Perturbation magnitude
        targeted: Whether to perform a targeted attack (default: False)
            - If False: maximize loss of current prediction (untargeted)
            - If True: minimize loss toward random class (targeted)
    
    Returns:
        Tuple of (adversarial_image, perturbation)
    """
    # Get original prediction
    original_seg = model.predict(image)
    
    # Get logits for the original prediction
    logits = model.get_logits(image)
    
    # Print shapes to help debug
    print(f"Original segmentation shape: {original_seg.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Create target labels based on attack type
    if targeted:
        # For targeted attack, create a random target different from current prediction
        n_classes = logits.shape[-1]
        # Create random target labels (different from current)
        target_labels = np.zeros_like(logits)
        
        # For each pixel, set a random target class that's different from original
        for h in range(original_seg.shape[0]):
            for w in range(original_seg.shape[1]):
                if h < logits.shape[1] and w < logits.shape[2]:  # Check bounds
                    current_class = original_seg[h, w]
                    # Choose any class except the current one
                    available_classes = list(range(n_classes))
                    if current_class < n_classes:  # Ensure current class is valid
                        available_classes.remove(current_class)
                    
                    if available_classes:  # If there are other classes available
                        target_class = np.random.choice(available_classes)
                        target_labels[0, h, w, target_class] = 1.0
                    else:
                        # Fallback if there's only one class
                        target_class = 0  # Default to class 0 if no valid classes
                        target_labels[0, h, w, target_class] = 1.0
    else:
        # For untargeted attack, use current prediction as target
        # (gradients will be negated later to maximize loss)
        target_labels = np.zeros_like(logits)
        
        # One-hot encode the current prediction
        for h in range(min(original_seg.shape[0], logits.shape[1])):
            for w in range(min(original_seg.shape[1], logits.shape[2])):
                class_idx = original_seg[h, w]
                if class_idx < logits.shape[3]:  # Ensure class index is valid
                    target_labels[0, h, w, class_idx] = 1.0
                else:
                    # If class index is out of range, use class 0
                    target_labels[0, h, w, 0] = 1.0
    
    # Compute gradients
    gradients = model.compute_gradients(image, target_labels)
    
    # For untargeted attack, negate the gradients to maximize loss
    if not targeted:
        gradients = -gradients
        
    # Create perturbation using the sign of gradients
    perturbation = epsilon * np.sign(gradients)
    
    # Ensure perturbation is in valid range
    perturbation = np.clip(perturbation, -epsilon, epsilon)
    
    # Create adversarial example
    adversarial_image = np.clip(image + perturbation, 0, 255).astype(np.uint8)
    
    # Return both the adversarial image and the perturbation
    return adversarial_image, perturbation[0]  # Remove batch dimension from perturbation


def calculate_prediction_difference(pred1, pred2):
    """
    Calculate the percentage of pixels that differ between two predictions.
    
    Args:
        pred1, pred2: Two prediction arrays of the same shape
        
    Returns:
        Percentage of pixels that differ
    """
    if pred1.shape != pred2.shape:
        raise ValueError(f"Predictions must have the same shape: {pred1.shape} vs {pred2.shape}")
    
    # Count differing pixels
    diff_pixels = np.sum(pred1 != pred2)
    total_pixels = pred1.size
    
    # Calculate percentage
    percentage = (diff_pixels / total_pixels) * 100
    
    return percentage


def generate_targeted_adversarial(image, model, target_class, epsilon=0.01):
    """
    Generate an adversarial example targeting a specific class using true FGSM.
    
    Args:
        image: Input image as numpy array
        model: DeepLabModel instance
        target_class: Target class to force the model to predict
        epsilon: Perturbation magnitude
        
    Returns:
        Tuple of (adversarial_image, perturbation)
    """
    # Get logits for the original prediction
    logits = model.get_logits(image)
    
    # Create target labels for the specified class
    target_labels = np.zeros_like(logits)
    target_labels[..., target_class] = 1.0
    
    # Compute gradients toward target class
    gradients = model.compute_gradients(image, target_labels)
    
    # Create perturbation using the sign of gradients (minimize loss toward target)
    perturbation = -epsilon * np.sign(gradients)
    
    # Ensure perturbation is in valid range
    perturbation = np.clip(perturbation, -epsilon, epsilon)
    
    # Create adversarial example
    adversarial_image = np.clip(image + perturbation, 0, 255).astype(np.uint8)
    
    # Return both the adversarial image and the perturbation
    return adversarial_image, perturbation[0]  # Remove batch dimension from perturbation


def batch_generate_adversarial(images, model, epsilon=0.01, targeted=False):
    """
    Generate adversarial examples for a batch of images using true FGSM.
    
    Args:
        images: List of input images as numpy arrays
        model: DeepLabModel instance
        epsilon: Perturbation magnitude
        targeted: Whether to perform targeted attacks
        
    Returns:
        List of tuples (adversarial_image, perturbation)
    """
    results = []
    for image in images:
        adversarial_image, perturbation = generate_adversarial_fgsm(
            image, model, epsilon, targeted
        )
        results.append((adversarial_image, perturbation))
    return results