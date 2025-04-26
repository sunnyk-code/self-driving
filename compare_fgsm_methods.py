#!/usr/bin/env python3
"""
Compare different FGSM implementations for semantic segmentation.
This script compares:
1. True FGSM using model gradients
2. Pattern-based FGSM (structured, edge, and frequency-based)
"""
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model_utils import DeepLabModel
from fast_fgsm import generate_fast_fgsm
from true_fgsm import generate_true_fgsm
from cityscapes_utils import create_colormap, CITYSCAPES_COLORS

def compare_fgsm_methods(image_path, epsilon=0.01, targeted=False, output_dir="fgsm_comparison_results"):
    """
    Compare different FGSM implementations on a single image.
    
    Args:
        image_path: Path to input image
        epsilon: Perturbation strength (default: 0.01)
        targeted: Whether to use targeted attack for true FGSM
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Comparing FGSM methods on {image_path} with epsilon={epsilon}")
    
    # Load image
    print("Loading image...")
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Load model
    print("Loading model...")
    model_path = os.path.join('models', 'deeplabv3_cityscapes_train', 'frozen_inference_graph.pb')
    model = DeepLabModel(model_path)
    
    # Get original prediction
    print("Running original prediction...")
    orig_pred = model.predict(image)
    
    # Methods to compare
    methods = {
        'true_fgsm': {'name': 'True FGSM (Gradient-based)', 'color': 'red'},
        'structured': {'name': 'Structured Pattern', 'color': 'blue'},
        'edge': {'name': 'Edge-based Pattern', 'color': 'green'},
        'frequency': {'name': 'Frequency-domain Pattern', 'color': 'purple'}
    }
    
    # Store results
    results = {}
    
    # Generate adversarial examples with each method
    for method_key, method_info in methods.items():
        print(f"\nGenerating adversarial example using {method_info['name']}...")
        
        start_time = time.time()
        
        if method_key == 'true_fgsm':
            # Use true FGSM with gradients
            adv_image, _ = generate_true_fgsm(image, model, epsilon, targeted, num_classes=19)
        else:
            # Use pattern-based methods
            adv_image = generate_fast_fgsm(image, epsilon, method_key)
        
        # Compute time taken
        elapsed = time.time() - start_time
        print(f"  Generation time: {elapsed:.2f} seconds")
        
        # Get prediction on adversarial example
        adv_pred = model.predict(adv_image)
        
        # Calculate differences
        diff_mask = orig_pred != adv_pred
        diff_percentage = np.sum(diff_mask) / np.prod(diff_mask.shape) * 100
        
        # Calculate perturbation
        perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
        perturbation_mean = np.mean(perturbation)
        perturbation_max = np.max(perturbation)
        
        # Store results
        results[method_key] = {
            'adv_image': adv_image,
            'adv_pred': adv_pred,
            'diff_percentage': diff_percentage,
            'generation_time': elapsed,
            'perturbation_mean': perturbation_mean,
            'perturbation_max': perturbation_max,
            'diff_mask': diff_mask
        }
        
        print(f"  Prediction difference: {diff_percentage:.2f}% of pixels changed")
        print(f"  Perturbation average: {perturbation_mean:.2f}, max: {perturbation_max:.2f}")
    
    # Colorize predictions
    orig_pred_color = create_colormap(orig_pred, CITYSCAPES_COLORS)
    for method_key in methods:
        results[method_key]['adv_pred_color'] = create_colormap(
            results[method_key]['adv_pred'], CITYSCAPES_COLORS
        )
    
    # Visualize results
    # 1. Create comparison of all methods (images and predictions)
    plt.figure(figsize=(15, 8 + 2 * len(methods)))
    
    # Original image and prediction
    plt.subplot(len(methods) + 1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(len(methods) + 1, 2, 2)
    plt.imshow(orig_pred_color)
    plt.title("Original Prediction")
    plt.axis('off')
    
    # Each method
    for i, (method_key, method_info) in enumerate(methods.items(), 1):
        result = results[method_key]
        
        # Adversarial image
        plt.subplot(len(methods) + 1, 2, i*2 + 1)
        plt.imshow(result['adv_image'])
        plt.title(f"{method_info['name']}\n(ε={epsilon}, time: {result['generation_time']:.2f}s)")
        plt.axis('off')
        
        # Adversarial prediction
        plt.subplot(len(methods) + 1, 2, i*2 + 2)
        plt.imshow(result['adv_pred_color'])
        plt.title(f"Changed: {result['diff_percentage']:.2f}%")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "method_comparison.png"))
    plt.close()
    
    # 2. Create comparison of perturbations
    plt.figure(figsize=(15, 8))
    
    # Original image
    plt.subplot(2, len(methods), 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Perturbations and difference masks
    for i, (method_key, method_info) in enumerate(methods.items(), 1):
        result = results[method_key]
        
        # Perturbation (amplified for visibility)
        plt.subplot(2, len(methods), i + 1)
        perturbation = np.abs(image.astype(np.float32) - result['adv_image'].astype(np.float32))
        perturbation = perturbation * 10  # Amplify
        perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)
        plt.imshow(perturbation, cmap='hot')
        plt.title(f"{method_info['name']}\nPerturbation (10x)")
        plt.axis('off')
        
        # Difference mask
        plt.subplot(2, len(methods), i + len(methods) + 1)
        diff_vis = np.zeros_like(image)
        diff_vis[..., 0] = 255 * result['diff_mask']  # Red channel
        plt.imshow(diff_vis)
        plt.title(f"Changed: {result['diff_percentage']:.2f}%")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perturbation_comparison.png"))
    plt.close()
    
    # 3. Create effectiveness comparison chart
    plt.figure(figsize=(10, 6))
    
    # Bar chart of effectiveness (% pixels changed)
    method_names = [methods[m]['name'] for m in methods]
    diff_percentages = [results[m]['diff_percentage'] for m in methods]
    colors = [methods[m]['color'] for m in methods]
    
    plt.bar(method_names, diff_percentages, color=colors)
    plt.ylabel('% Pixels Changed in Prediction')
    plt.title(f'Effectiveness of Different FGSM Methods (ε={epsilon})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effectiveness_comparison.png"))
    plt.close()
    
    # 4. Create generation time comparison chart
    plt.figure(figsize=(10, 6))
    
    # Bar chart of generation times
    gen_times = [results[m]['generation_time'] for m in methods]
    
    plt.bar(method_names, gen_times, color=colors)
    plt.ylabel('Generation Time (seconds)')
    plt.title(f'Generation Time of Different FGSM Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()
    
    # 5. Create efficiency comparison (effectiveness/time)
    plt.figure(figsize=(10, 6))
    
    # Bar chart of efficiency
    efficiency = [diff_percentages[i]/max(0.01, gen_times[i]) for i in range(len(methods))]
    
    plt.bar(method_names, efficiency, color=colors)
    plt.ylabel('Efficiency (% Changed / Second)')
    plt.title(f'Efficiency of Different FGSM Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_comparison.png"))
    plt.close()
    
    # Save individual images
    for method_key in methods:
        Image.fromarray(results[method_key]['adv_image']).save(
            os.path.join(output_dir, f"adv_{method_key}_eps_{epsilon}.png")
        )
    
    # Save summary report
    with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
        f.write(f"FGSM Methods Comparison Results\n")
        f.write(f"==============================\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"True FGSM attack type: {'Targeted' if targeted else 'Untargeted'}\n\n")
        
        f.write(f"Effectiveness (% pixels changed):\n")
        for method_key, method_info in methods.items():
            f.write(f"  {method_info['name']}: {results[method_key]['diff_percentage']:.2f}%\n")
        
        f.write(f"\nGeneration time (seconds):\n")
        for method_key, method_info in methods.items():
            f.write(f"  {method_info['name']}: {results[method_key]['generation_time']:.2f}s\n")
        
        f.write(f"\nPerturbation statistics:\n")
        for method_key, method_info in methods.items():
            f.write(f"  {method_info['name']}:\n")
            f.write(f"    - Mean: {results[method_key]['perturbation_mean']:.2f}\n")
            f.write(f"    - Max: {results[method_key]['perturbation_max']:.2f}\n")
    
    # Clean up
    model.close()
    
    print(f"\nComparison results saved to {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare FGSM implementations")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Perturbation strength (default: 0.01)")
    parser.add_argument("--targeted", action="store_true",
                        help="Use targeted attack for true FGSM (default: False)")
    parser.add_argument("--output_dir", type=str, default="fgsm_comparison_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    compare_fgsm_methods(args.image_path, args.epsilon, args.targeted, args.output_dir)

if __name__ == "__main__":
    main() 