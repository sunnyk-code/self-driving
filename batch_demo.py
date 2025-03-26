#!/usr/bin/env python3
"""
Batch Demo Script for Cityscapes Segmentation and Adversarial Examples
"""
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

from main import process_single_example, DEFAULT_EPSILON
from cityscapes_utils import get_dataset_files, create_colormap
from visualization import visualize_results

# Constants
RESULTS_DIR = 'batch_results'

def create_summary_grid(results, output_dir, num_images=5, title="Batch Results Summary"):
    """
    Create a summary grid visualization of batch results.
    
    Args:
        results: List of result dictionaries from process_single_example
        output_dir: Directory to save the summary grid
        num_images: Number of images to include in the grid (will use first N)
        title: Title for the summary visualization
    """
    if not results:
        print("No results to visualize!")
        return
    
    # Limit to the specified number of images
    results = results[:num_images]
    n_results = len(results)
    
    # Create figure
    fig, axes = plt.subplots(n_results, 5, figsize=(20, 4*n_results))
    
    # If only one result, make axes indexable
    if n_results == 1:
        axes = [axes]
    
    # Loop through results
    for i, result in enumerate(results):
        # Original image
        axes[i, 0].imshow(result['original_image'])
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].axis('off')
        
        # Ground truth (if available)
        if result['ground_truth'] is not None:
            axes[i, 1].imshow(result['ground_truth'], cmap='nipy_spectral')
            axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        else:
            axes[i, 1].axis('off')
            axes[i, 1].set_title("No Ground Truth" if i == 0 else "")
        axes[i, 1].axis('off')
        
        # Original prediction
        axes[i, 2].imshow(result['original_prediction'], cmap='nipy_spectral')
        axes[i, 2].set_title("Original Prediction" if i == 0 else "")
        axes[i, 2].axis('off')
        
        # Adversarial image
        axes[i, 3].imshow(result['adversarial_image'])
        axes[i, 3].set_title("Adversarial Image" if i == 0 else "")
        axes[i, 3].axis('off')
        
        # Adversarial prediction
        axes[i, 4].imshow(result['adversarial_prediction'], cmap='nipy_spectral')
        axes[i, 4].set_title(f"Adversarial Prediction\nDiff: {result['diff_percentage']:.2f}%" if i == 0 else f"Diff: {result['diff_percentage']:.2f}%")
        axes[i, 4].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the summary grid
    summary_path = os.path.join(output_dir, "batch_summary.png")
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    print(f"Saved batch summary to: {summary_path}")

def main():
    """Main function to parse arguments and run the batch processing."""
    parser = argparse.ArgumentParser(description='Batch Processing for Cityscapes Segmentation and Adversarial Examples')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing Cityscapes dataset')
    parser.add_argument('--subset', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset subset to use (default: val)')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to process (default: 5)')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON,
                        help='Epsilon value for FGSM attack (default: 0.01)')
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR,
                        help=f'Directory to save results (default: {RESULTS_DIR})')
    parser.add_argument('--city', type=str, default=None,
                        help='Specific city to use (optional)')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable individual visualization generation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of files to process
    images, labels = get_dataset_files(args.data_dir, args.subset, args.city)
    
    if not images:
        print(f"No images found in {args.data_dir} for subset {args.subset}" + 
              (f" and city {args.city}" if args.city else ""))
        return
    
    print(f"Found {len(images)} images")
    
    # Limit the number of images to process
    if args.num_images > 0 and args.num_images < len(images):
        # Randomly select num_images from the dataset
        indices = np.random.choice(len(images), args.num_images, replace=False)
        selected_images = [images[i] for i in indices]
        selected_labels = [labels[i] for i in indices] if labels else [None] * len(selected_images)
    else:
        selected_images = images
        selected_labels = labels if labels else [None] * len(selected_images)
    
    print(f"Selected {len(selected_images)} images for processing")
    
    # Process each image
    results = []
    start_time = time.time()
    
    for i, (image_path, label_path) in enumerate(zip(selected_images, selected_labels)):
        print(f"\nProcessing image {i+1}/{len(selected_images)}: {os.path.basename(image_path)}")
        
        # Create subfolder for this image
        image_output_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_path))[0])
        
        # Process the example
        result = process_single_example(
            image_path=image_path,
            label_path=label_path,
            epsilon=args.epsilon,
            output_dir=image_output_dir,
            visualize=not args.no_visualize
        )
        
        results.append(result)
    
    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nProcessed {len(results)} images in {elapsed_time:.2f} seconds "
          f"({elapsed_time/len(results):.2f} seconds per image)")
    
    # Create summary visualization
    create_summary_grid(results, args.output_dir, 
                        num_images=min(5, len(results)), 
                        title=f"Batch Results (ε={args.epsilon}, {args.subset} set)")
    
    # Calculate and print average difference percentage
    avg_diff = np.mean([r['diff_percentage'] for r in results])
    print(f"Average prediction difference: {avg_diff:.2f}%")
    
    # Save summary statistics
    with open(os.path.join(args.output_dir, "batch_summary.txt"), "w") as f:
        f.write(f"Batch Processing Summary\n")
        f.write(f"=======================\n")
        f.write(f"Dataset: {args.data_dir}, Subset: {args.subset}, City: {args.city or 'all'}\n")
        f.write(f"Epsilon: {args.epsilon}\n")
        f.write(f"Number of images processed: {len(results)}\n")
        f.write(f"Processing time: {elapsed_time:.2f} seconds ({elapsed_time/len(results):.2f} seconds per image)\n")
        f.write(f"Average prediction difference: {avg_diff:.2f}%\n\n")
        
        f.write(f"Individual Results:\n")
        for i, result in enumerate(results):
            img_name = os.path.basename(selected_images[i])
            f.write(f"{i+1}. {img_name}: Diff={result['diff_percentage']:.2f}%\n")
    
    print(f"Saved summary statistics to: {os.path.join(args.output_dir, 'batch_summary.txt')}")

if __name__ == "__main__":
    main() 