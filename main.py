#!/usr/bin/env python3
"""
Main script for Cityscapes Segmentation and Adversarial Examples
"""
import argparse
import os
import numpy as np
from PIL import Image

# Import our modules
from cityscapes_utils import create_colormap, find_label_path, get_dataset_files
from model_utils import DeepLabModel, extract_model
from adversarial import generate_adversarial_fgsm, calculate_prediction_difference
from visualization import visualize_results, visualize_perturbation

# Constants
MODEL_URL = 'http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz'
MODEL_DIR = 'models'
MODEL_FILENAME = 'deeplabv3_cityscapes_train_2018_02_06.tar.gz'
RESULTS_DIR = 'results'
DEFAULT_EPSILON = 0.01  # Default epsilon for FGSM attack

def load_image(image_path):
    """Load an image as RGB numpy array."""
    try:
        img = np.array(Image.open(image_path).convert('RGB'))
        return img
    except Exception as e:
        raise ValueError(f"Error loading image from {image_path}: {e}")

def load_label(label_path):
    """Load a label image as numpy array."""
    try:
        if label_path is None:
            return None
        label = np.array(Image.open(label_path))
        # If label has 3 channels, it's likely a color image
        if len(label.shape) == 3 and label.shape[2] == 3:
            # Convert RGB to grayscale for labels
            label = np.array(Image.open(label_path).convert('L'))
        return label
    except Exception as e:
        raise ValueError(f"Error loading label from {label_path}: {e}")

def process_single_example(image_path, label_path=None, epsilon=DEFAULT_EPSILON, output_dir=RESULTS_DIR, 
                          colormap=None, visualize=True, targeted=False):
    """
    Process a single image through the entire pipeline: 
    prediction, adversarial generation, evaluation.
    
    Args:
        image_path: Path to the input image
        label_path: Path to the ground truth label (optional)
        epsilon: Strength of the adversarial perturbation
        output_dir: Directory to save results
        colormap: Color mapping for visualization
        visualize: Whether to create and save visualizations
        targeted: Whether to perform a targeted attack (default: False)
    
    Returns:
        Dict containing results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing image: {image_path}")
    if label_path:
        print(f"Ground truth label: {label_path}")
    else:
        print("No ground truth label provided")
        
    # Extract model if necessary and load it
    model_path = os.path.join(MODEL_DIR, 'deeplabv3_cityscapes_train', 'frozen_inference_graph.pb')
    if not os.path.exists(model_path):
        print(f"Extracting model to {MODEL_DIR}...")
        extract_model(MODEL_FILENAME, MODEL_DIR)
    
    print(f"Loading model from {model_path}...")
    model = DeepLabModel(model_path)
    
    # Load the image
    original_image = load_image(image_path)
    print(f"Loaded image with shape: {original_image.shape}")
    
    # Load ground truth if available
    ground_truth = None
    if label_path:
        ground_truth = load_label(label_path)
        print(f"Loaded ground truth with shape: {ground_truth.shape}")
    
    # Get original prediction
    print("Generating original prediction...")
    original_seg = model.predict(original_image)
    print(f"Original prediction shape: {original_seg.shape}")
    
    # Generate adversarial example using proper FGSM
    print(f"Generating adversarial example with epsilon={epsilon} using FGSM...")
    adversarial_image, _ = generate_adversarial_fgsm(original_image, model, epsilon, targeted=targeted)
    print(f"Adversarial image shape: {adversarial_image.shape}")
    
    # Get prediction on adversarial example
    print("Generating prediction on adversarial example...")
    adversarial_seg = model.predict(adversarial_image)
    print(f"Adversarial prediction shape: {adversarial_seg.shape}")
    
    # Calculate difference between predictions
    diff_percentage = calculate_prediction_difference(original_seg, adversarial_seg)
    print(f"Percentage of pixels that changed in prediction: {diff_percentage:.2f}%")
    
    # Generate base filename for outputs
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save results
    if visualize:
        # Visualize results
        print("Generating visualizations...")
        
        # Create colorized versions of segmentations
        if colormap is not None:
            original_seg_vis = create_colormap(original_seg, colormap)
            adversarial_seg_vis = create_colormap(adversarial_seg, colormap)
            
            if ground_truth is not None:
                ground_truth_vis = create_colormap(ground_truth, colormap)
            else:
                ground_truth_vis = None
        else:
            original_seg_vis = original_seg
            adversarial_seg_vis = adversarial_seg
            ground_truth_vis = ground_truth
        
        # Save full comparison
        vis_path = os.path.join(output_dir, f"{base_filename}_comparison.png")
        visualize_results(
            original_image=original_image, 
            ground_truth=ground_truth_vis, 
            original_pred=original_seg_vis,
            adversarial_image=adversarial_image, 
            adversarial_pred=adversarial_seg_vis,
            title=f"Segmentation Results (ε={epsilon}, diff={diff_percentage:.2f}%)",
            save_path=vis_path
        )
        print(f"Saved comparison visualization to: {vis_path}")
        
        # Save perturbation visualization
        pert_path = os.path.join(output_dir, f"{base_filename}_perturbation.png")
        visualize_perturbation(
            original_image=original_image,
            adversarial_image=adversarial_image,
            save_path=pert_path
        )
        print(f"Saved perturbation visualization to: {pert_path}")
    
    # Save individual images
    adv_img_path = os.path.join(output_dir, f"{base_filename}_adversarial.png")
    Image.fromarray(adversarial_image).save(adv_img_path)
    print(f"Saved adversarial image to: {adv_img_path}")
    
    # Cleanup
    model.close()
    
    return {
        "original_image": original_image,
        "ground_truth": ground_truth,
        "original_prediction": original_seg,
        "adversarial_image": adversarial_image,
        "adversarial_prediction": adversarial_seg,
        "diff_percentage": diff_percentage
    }

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Cityscapes Segmentation and Adversarial Examples')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--label_path', type=str, default=None,
                        help='Path to ground truth label (optional)')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON,
                        help='Epsilon value for FGSM attack (default: 0.01)')
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR,
                        help=f'Directory to save results (default: {RESULTS_DIR})')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization generation')
    parser.add_argument('--targeted', action='store_true',
                        help='Use targeted FGSM attack (default: False)')
    args = parser.parse_args()
    
    # Auto-detect label path if not provided
    if args.label_path is None:
        args.label_path = find_label_path(args.image_path)
        if args.label_path:
            print(f"Auto-detected label path: {args.label_path}")
    
    # Process the example
    process_single_example(
        image_path=args.image_path,
        label_path=args.label_path,
        epsilon=args.epsilon,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        targeted=args.targeted
    )

if __name__ == "__main__":
    main() 