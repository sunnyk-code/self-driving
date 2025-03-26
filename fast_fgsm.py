#!/usr/bin/env python3
"""
Fast FGSM-like attack for semantic segmentation models.
This uses structured noise patterns that are effective against
segmentation models without needing expensive gradient computations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

def generate_fast_fgsm(image, epsilon=0.01, pattern_type='structured'):
    """
    Generate an adversarial example using a fast FGSM-like method
    with structured noise patterns designed to affect semantic segmentation.
    
    Args:
        image: Input image as numpy array (uint8)
        epsilon: Strength of the perturbation (0 to 1)
        pattern_type: Type of structured noise ('structured', 'edge', 'frequency')
        
    Returns:
        Adversarial example as numpy array (uint8)
    """
    # Convert to float32 for manipulation
    image_float = image.astype(np.float32)
    h, w, c = image.shape
    
    # Choose perturbation pattern based on type
    if pattern_type == 'edge':
        # Edge-based perturbation (high-pass filter)
        from scipy.ndimage import sobel
        
        # Calculate edge map for each channel
        edge_x = np.zeros_like(image_float)
        edge_y = np.zeros_like(image_float)
        
        for i in range(3):
            edge_x[..., i] = sobel(image_float[..., i], axis=1)
            edge_y[..., i] = sobel(image_float[..., i], axis=0)
            
        # Combine edge maps
        perturbation = np.sqrt(edge_x**2 + edge_y**2)
        
    elif pattern_type == 'frequency':
        # Frequency-based perturbation using DCT
        try:
            from scipy.fft import dctn, idctn
        except ImportError:
            # Fallback for older SciPy versions
            from scipy.fftpack import dctn, idctn
        
        # Compute DCT of each channel
        dct = np.zeros_like(image_float)
        for i in range(3):
            dct[..., i] = dctn(image_float[..., i])
        
        # Create a mask that emphasizes mid-frequency components
        # (these are important for object recognition)
        mask = np.ones((h, w))
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - h/2)**2 + (x - w/2)**2)
        
        # Attenuate low and high frequencies
        mask[r < min(h, w)/8] = 0.2  # Low frequencies
        mask[r > min(h, w)/2] = 0.2  # High frequencies
        
        # Apply mask to DCT coefficients
        for i in range(3):
            dct[..., i] *= mask
        
        # Inverse DCT
        perturbation = np.zeros_like(image_float)
        for i in range(3):
            perturbation[..., i] = idctn(dct[..., i])
            
    else:  # Default: structured pattern
        # Create structured patterns known to affect segmentation models
        # Pattern 1: Grid pattern
        grid_size = 16
        grid = np.zeros((h, w))
        grid[::grid_size, :] = 1.0
        grid[:, ::grid_size] = 1.0
        
        # Pattern 2: Radial pattern
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h/2, w/2
        r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        radial = np.sin(r / 20) * 0.5 + 0.5
        
        # Pattern 3: Random noise with spatial correlation
        from scipy.ndimage import gaussian_filter
        noise = np.random.randn(h, w)
        smooth_noise = gaussian_filter(noise, sigma=5.0)
        smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
        
        # Combine patterns with different weights for each channel
        perturbation = np.zeros_like(image_float)
        perturbation[..., 0] = grid * 0.5 + smooth_noise * 0.5  # Red channel
        perturbation[..., 1] = radial * 0.6 + smooth_noise * 0.4  # Green channel
        perturbation[..., 2] = smooth_noise  # Blue channel
    
    # Normalize perturbation to have values between -1 and 1
    perturbation = 2.0 * (perturbation - np.min(perturbation)) / (np.max(perturbation) - np.min(perturbation)) - 1.0
    
    # Add the perturbation to the image using FGSM formula: image + epsilon * sign(perturbation)
    perturbed_image = image_float + epsilon * 255.0 * np.sign(perturbation)
    
    # Ensure valid image by clipping to [0, 255]
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    
    return perturbed_image

def compare_fast_fgsm_methods(image_path, epsilon=0.01, output_dir='fast_fgsm_results'):
    """
    Compare different fast FGSM methods on a single image.
    
    Args:
        image_path: Path to input image
        epsilon: Perturbation strength
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Apply different attack methods
    methods = ['structured', 'edge', 'frequency']
    perturbed_images = {}
    times = {}
    
    for method in methods:
        print(f"Generating adversarial example using {method} pattern...")
        start_time = time.time()
        perturbed = generate_fast_fgsm(image, epsilon, method)
        elapsed = time.time() - start_time
        
        perturbed_images[method] = perturbed
        times[method] = elapsed
        
        print(f"  Time taken: {elapsed:.2f} seconds")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(len(methods) + 1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Empty plots for alignment
    plt.subplot(len(methods) + 1, 3, 2)
    plt.axis('off')
    plt.subplot(len(methods) + 1, 3, 3)
    plt.axis('off')
    
    # Different methods
    for i, method in enumerate(methods):
        # Perturbed image
        plt.subplot(len(methods) + 1, 3, (i+1)*3 + 1)
        plt.imshow(perturbed_images[method])
        plt.title(f"{method.capitalize()} (ε={epsilon})")
        plt.axis('off')
        
        # Perturbation
        plt.subplot(len(methods) + 1, 3, (i+1)*3 + 2)
        diff = np.abs(image.astype(np.float32) - perturbed_images[method].astype(np.float32))
        # Amplify for visibility
        diff = diff * 10
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        plt.imshow(diff)
        plt.title(f"Perturbation (10x)")
        plt.axis('off')
        
        # Colormap for better visibility
        plt.subplot(len(methods) + 1, 3, (i+1)*3 + 3)
        plt.imshow(diff, cmap='viridis')
        plt.title(f"Perturbation (colormap)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_epsilon_{epsilon}.png"))
    plt.close()
    
    # Save individual images
    for method in methods:
        Image.fromarray(perturbed_images[method]).save(
            os.path.join(output_dir, f"adv_{method}_epsilon_{epsilon}.png")
        )
    
    # Save timing information
    with open(os.path.join(output_dir, "timing_info.txt"), "w") as f:
        f.write(f"Timing Information (epsilon={epsilon})\n")
        f.write("===============================\n")
        for method in methods:
            f.write(f"{method.capitalize()}: {times[method]:.2f} seconds\n")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast FGSM-like attacks for semantic segmentation")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Perturbation strength (default: 0.01)")
    parser.add_argument("--output_dir", type=str, default="fast_fgsm_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    compare_fast_fgsm_methods(args.image_path, args.epsilon, args.output_dir) 