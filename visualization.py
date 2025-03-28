import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def visualize_results(original_image, ground_truth=None, original_pred=None, 
                      adversarial_image=None, adversarial_pred=None, 
                      colormap=None, title="Segmentation Results", 
                      save_path=None, figsize=(20, 10)):
    """
    Visualize segmentation results including original and adversarial images and predictions.
    
    Args:
        original_image: Original input image
        ground_truth: Ground truth segmentation mask (optional)
        original_pred: Model prediction on original image (optional)
        adversarial_image: Adversarial example (optional)
        adversarial_pred: Model prediction on adversarial image (optional)
        colormap: Colormap for segmentation masks (if None, will use random colors)
        title: Plot title
        save_path: Path to save visualization (if None, will display)
        figsize: Figure size
    """
    # Determine how many images to display based on what's provided
    n_cols = sum(x is not None for x in [original_image, ground_truth, original_pred, 
                                        adversarial_image, adversarial_pred])
    
    if n_cols == 0:
        raise ValueError("At least one image must be provided")
    
    # Create figure
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]  # Make axes indexable for the single-image case
    
    # Plot original image
    col_idx = 0
    if original_image is not None:
        axes[col_idx].imshow(original_image)
        axes[col_idx].set_title("Original Image")
        axes[col_idx].axis('off')
        col_idx += 1
    
    # Plot ground truth if provided
    if ground_truth is not None:
        # Don't need to explicitly convert to ListedColormap here
        axes[col_idx].imshow(ground_truth, cmap=colormap)
        axes[col_idx].set_title("Ground Truth")
        axes[col_idx].axis('off')
        col_idx += 1
    
    # Plot original prediction if provided
    if original_pred is not None:
        # Don't need to explicitly convert to ListedColormap here
        axes[col_idx].imshow(original_pred, cmap=colormap)
        axes[col_idx].set_title("Original Prediction")
        axes[col_idx].axis('off')
        col_idx += 1
    
    # Plot adversarial image if provided
    if adversarial_image is not None:
        axes[col_idx].imshow(adversarial_image)
        axes[col_idx].set_title("Adversarial Image")
        axes[col_idx].axis('off')
        col_idx += 1
    
    # Plot adversarial prediction if provided
    if adversarial_pred is not None:
        # Don't need to explicitly convert to ListedColormap here
        axes[col_idx].imshow(adversarial_pred, cmap=colormap)
        axes[col_idx].set_title("Adversarial Prediction")
        axes[col_idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save or display
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def visualize_perturbation(original_image, adversarial_image, save_path=None, figsize=(15, 5)):
    """
    Visualize the perturbation applied to an image to create an adversarial example.
    
    Args:
        original_image: Original image
        adversarial_image: Adversarial image
        save_path: Path to save visualization (if None, will display)
        figsize: Figure size
    """
    # Calculate perturbation (difference)
    perturbation = adversarial_image.astype(np.float32) - original_image.astype(np.float32)
    
    # Scale perturbation for better visualization
    # Add 128 to center at gray, so zero perturbation is gray
    perturbation_vis = np.clip(perturbation + 128, 0, 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Perturbation
    axes[1].imshow(perturbation_vis)
    axes[1].set_title("Perturbation\n(amplified for visibility)")
    axes[1].axis('off')
    
    # Adversarial image
    axes[2].imshow(adversarial_image)
    axes[2].set_title("Adversarial Image")
    axes[2].axis('off')
    
    plt.suptitle(f"Perturbation Analysis")
    plt.tight_layout()
    
    # Save or display
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_perturbation_only(perturbation, save_path=None, figsize=(10, 5)):
    """
    Visualize the perturbation directly.
    
    Args:
        perturbation: Perturbation array (difference between adversarial and original)
        save_path: Path to save visualization (if None, will display)
        figsize: Figure size
        
    Returns:
        Visualization of perturbation as a numpy array
    """
    # Scale perturbation for better visualization
    # Map values to range [0, 255]
    abs_pert = np.abs(perturbation)
    if abs_pert.max() > 0:  # Avoid division by zero
        # Scale to [0, 255] based on max magnitude
        scale_factor = 255.0 / abs_pert.max()
    else:
        scale_factor = 1.0
        
    # Create a RGB visualization where:
    # - Red: positive perturbation
    # - Blue: negative perturbation
    # - Intensity: magnitude of perturbation
    pert_vis = np.zeros((*perturbation.shape[0:2], 3), dtype=np.uint8)
    
    # Red channel: positive perturbations
    pos_mask = perturbation > 0
    if np.any(pos_mask):
        for c in range(3):  # Apply to all channels
            pert_vis[:, :, 0][pos_mask[:, :, c]] = np.clip(
                perturbation[:, :, c][pos_mask[:, :, c]] * scale_factor, 
                0, 255
            ).astype(np.uint8)
    
    # Blue channel: negative perturbations
    neg_mask = perturbation < 0
    if np.any(neg_mask):
        for c in range(3):  # Apply to all channels
            pert_vis[:, :, 2][neg_mask[:, :, c]] = np.clip(
                -perturbation[:, :, c][neg_mask[:, :, c]] * scale_factor, 
                0, 255
            ).astype(np.uint8)
    
    # If save_path is provided, save the visualization
    if save_path:
        plt.figure(figsize=figsize)
        plt.imshow(pert_vis)
        plt.title("Perturbation (Red: positive, Blue: negative)")
        plt.axis('off')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    return pert_vis