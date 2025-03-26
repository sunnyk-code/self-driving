#!/usr/bin/env python3
"""
Example script demonstrating how to use the Cityscapes segmentation and adversarial example tools.
"""
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# Import from our modules
from main import process_single_example
from model_utils import DeepLabModel, extract_model
from cityscapes_utils import create_colormap, CITYSCAPES_COLORS, load_cityscapes_image, load_cityscapes_label
from adversarial import generate_adversarial_fgsm, calculate_prediction_difference
from visualization import visualize_results, visualize_perturbation

def generate_random_adversarial(image, model, epsilon=0.01):
    """Generate an adversarial example using random noise (for comparison)."""
    # Convert to float32 for manipulation
    image_float = image.astype(np.float32)
    
    # Get original prediction
    orig_pred = model.predict(image)
    
    # Random noise perturbation
    noise = np.random.uniform(-1, 1, size=image.shape)
    
    # Scale noise by epsilon and add to image
    perturbed_image = image_float + epsilon * 255.0 * np.sign(noise)
    
    # Ensure valid image by clipping to [0, 255]
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    
    return perturbed_image, orig_pred

def compare_methods(image_path, model, epsilon=0.01, output_dir="comparison_results"):
    """Compare FGSM and random noise perturbation methods."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nComparing FGSM with random noise perturbation on {os.path.basename(image_path)}...")
    
    # Load image
    original_image = load_cityscapes_image(image_path)
    
    # Generate predictions and adversarial examples
    original_pred = model.predict(original_image)
    
    # Generate adversarial examples with FGSM
    print("\nGenerating adversarial example with FGSM...")
    fgsm_image, _ = generate_adversarial_fgsm(original_image, model, epsilon)
    fgsm_pred = model.predict(fgsm_image)
    fgsm_diff = calculate_prediction_difference(original_pred, fgsm_pred)
    print(f"FGSM: Changed {fgsm_diff:.2f}% of pixels in prediction")
    
    # Generate adversarial examples with random noise (for comparison)
    print("\nGenerating adversarial example with random noise (for comparison)...")
    random_image, _ = generate_random_adversarial(original_image, model, epsilon)
    random_pred = model.predict(random_image)
    random_diff = calculate_prediction_difference(original_pred, random_pred)
    print(f"Random noise: Changed {random_diff:.2f}% of pixels in prediction")
    
    # Create colorized versions of segmentations
    original_seg_vis = create_colormap(original_pred, CITYSCAPES_COLORS)
    fgsm_seg_vis = create_colormap(fgsm_pred, CITYSCAPES_COLORS)
    random_seg_vis = create_colormap(random_pred, CITYSCAPES_COLORS)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(original_seg_vis)
    plt.title("Original Prediction")
    plt.axis('off')
    
    # FGSM
    plt.subplot(2, 3, 2)
    plt.imshow(fgsm_image)
    plt.title(f"FGSM Adversarial\n(ε={epsilon})")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(fgsm_seg_vis)
    plt.title(f"FGSM Prediction\nDiff: {fgsm_diff:.2f}%")
    plt.axis('off')
    
    # Random
    plt.subplot(2, 3, 3)
    plt.imshow(random_image)
    plt.title(f"Random Noise\n(ε={epsilon})")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(random_seg_vis)
    plt.title(f"Random Noise Prediction\nDiff: {random_diff:.2f}%")
    plt.axis('off')
    
    plt.suptitle("Comparison: FGSM vs Random Noise Perturbation", fontsize=16)
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, "fgsm_vs_random.png")
    plt.savefig(comparison_path)
    plt.close()
    print(f"Saved comparison visualization to: {comparison_path}")
    
    # Also save the FGSM perturbation visualization
    visualize_perturbation(
        original_image=original_image,
        adversarial_image=fgsm_image,
        save_path=os.path.join(output_dir, "fgsm_perturbation.png")
    )
    
    return fgsm_diff, random_diff

def main():
    """
    Main function to demonstrate the project usage.
    """
    print("Cityscapes Segmentation and Adversarial Examples - Example Script")
    print("===============================================================")
    
    # Define default paths
    model_filename = "deeplabv3_cityscapes_train_2018_02_06.tar.gz"
    model_dir = "models"
    results_dir = "example_results"
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Ensure model is available
    print("\nStep 1: Checking for DeepLabV3 model...")
    model_path = os.path.join(model_dir, "deeplabv3_cityscapes_train", "frozen_inference_graph.pb")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        if os.path.exists(model_filename):
            print(f"Extracting model from {model_filename}...")
            extract_model(model_filename, model_dir)
        else:
            print(f"Please download the model file {model_filename} from TensorFlow model zoo:")
            print("http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz")
            return
    else:
        print(f"Model found at {model_path}")
    
    # Step 2: Process an example image
    print("\nStep 2: Looking for an example image to process...")
    
    # Try to find an example image in the data directory
    sample_image_path = None
    possible_paths = [
        "data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_008688_leftImg8bit.png",
        "data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000001_008688_leftImg8bit.png"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sample_image_path = path
            break
    
    # If no sample image found, use a placeholder
    if sample_image_path is None:
        print("No sample Cityscapes image found. Using a placeholder image...")
        # Create a simple placeholder image
        placeholder = Image.new('RGB', (512, 256), color='gray')
        # Add some colored shapes for demonstration
        from PIL import ImageDraw
        draw = ImageDraw.Draw(placeholder)
        draw.rectangle([(50, 50), (200, 150)], fill='blue')  # building
        draw.rectangle([(0, 180), (512, 256)], fill='green')  # road
        draw.ellipse([(300, 50), (400, 150)], fill='red')    # car
        
        sample_image_path = os.path.join(results_dir, "placeholder.png")
        placeholder.save(sample_image_path)
    
    print(f"Using image: {sample_image_path}")
    
    # Step 3: Load the model for direct access
    print("\nStep 3: Loading DeepLabV3 model...")
    model = DeepLabModel(model_path)
    
    # Step 4: Compare FGSM with random noise perturbation
    print("\nStep 4: Comparing FGSM with random noise perturbation...")
    comparison_dir = os.path.join(results_dir, "comparison")
    fgsm_diff, random_diff = compare_methods(
        image_path=sample_image_path,
        model=model,
        epsilon=0.01,
        output_dir=comparison_dir
    )
    
    # Step 5: Process the image with different epsilon values
    print("\nStep 5: Processing image with different epsilon values using proper FGSM...")
    
    epsilon_values = [0.01, 0.03, 0.05]
    fgsm_results = []
    
    for epsilon in epsilon_values:
        print(f"\nProcessing with epsilon = {epsilon}...")
        output_dir = os.path.join(results_dir, f"epsilon_{epsilon:.2f}")
        
        # Process the image
        result = process_single_example(
            image_path=sample_image_path,
            label_path=None,  # Auto-detect label if available
            epsilon=epsilon,
            output_dir=output_dir,
            colormap=CITYSCAPES_COLORS,
            visualize=True
        )
        
        fgsm_results.append(result['diff_percentage'])
        print(f"Percentage of pixels changed in prediction: {result['diff_percentage']:.2f}%")
    
    # Step 6: Plot comparison of effectiveness by epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, fgsm_results, 'o-', label='FGSM')
    plt.title('Effect of Epsilon on Prediction Changes')
    plt.xlabel('Epsilon')
    plt.ylabel('Percentage of Changed Pixels')
    plt.grid(True)
    plt.legend()
    
    effectiveness_path = os.path.join(results_dir, "epsilon_effectiveness.png")
    plt.savefig(effectiveness_path)
    plt.close()
    print(f"\nSaved effectiveness comparison to: {effectiveness_path}")
    
    # Clean up
    model.close()
    
    print("\nExample completed! Results are saved in the 'example_results' directory.")
    print("\nKey findings:")
    print(f"1. FGSM changed {fgsm_diff:.2f}% of pixels in the prediction")
    print(f"2. Random noise changed {random_diff:.2f}% of pixels in the prediction")
    print(f"3. FGSM is {fgsm_diff/random_diff:.1f}x more effective than random noise at the same epsilon value!")
    
    print("\nTo process your own images, use commands like:")
    print("\npython main.py --image_path your_image.png --epsilon 0.01")
    print("\nFor batch processing use:")
    print("python batch_demo.py --data_dir path/to/cityscapes --num_images 5 --epsilon 0.01")

if __name__ == "__main__":
    main() 