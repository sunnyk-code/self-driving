#!/usr/bin/env python3
"""
Fixed implementation of Fast Gradient Sign Method (FGSM) for semantic segmentation.
This implementation uses TensorFlow eager execution to compute gradients reliably.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import time

# Enable eager execution
tf.compat.v1.enable_eager_execution()

def create_tf_model(model_path):
    """
    Create a TensorFlow model for gradient computation.
    
    Args:
        model_path: Path to the frozen inference graph
        
    Returns:
        Model function that takes an input image and returns logits
    """
    # Load the frozen graph
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    def model_fn(input_image):
        """Model function that returns logits given input image."""
        # Create a tensor from the input image
        # The image should be in format [batch, height, width, channels]
        if len(input_image.shape) == 3:
            input_image = tf.expand_dims(input_image, 0)
        
        # Import the graph and get logits tensor
        # For DeepLabV3, the logits are at 'ResizeBilinear_2:0'
        logits = tf.compat.v1.import_graph_def(
            graph_def,
            input_map={"ImageTensor:0": input_image},
            return_elements=["ResizeBilinear_2:0"],
            name=""
        )[0]
        
        return logits
    
    return model_fn

def generate_fgsm_with_eager(image_path, model_path, epsilon=0.01, targeted=False, num_classes=19, output_dir='fixed_fgsm_results'):
    """
    Generate adversarial example using FGSM with TensorFlow eager execution.
    
    Args:
        image_path: Path to input image
        model_path: Path to frozen model
        epsilon: Perturbation strength (default: 0.01)
        targeted: Whether to use targeted attack (default: False)
        num_classes: Number of classes in the model (default: 19)
        output_dir: Directory to save results (default: 'fixed_fgsm_results')
        
    Returns:
        Path to the output visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    print(f"Loading image: {image_path}")
    # Load and prepare the image
    image = np.array(Image.open(image_path).convert('RGB'))
    input_image = tf.cast(image, tf.float32)
    
    # Create a model function
    print(f"Loading model from: {model_path}")
    model_fn = create_tf_model(model_path)
    
    # Create a DeepLabModel for getting the original prediction
    from model_utils import DeepLabModel
    deeplab_model = DeepLabModel(model_path)
    
    # Get original prediction
    print("Getting original prediction...")
    orig_pred = deeplab_model.predict(image)
    
    # Create a GradientTape context to track operations for automatic differentiation
    print(f"Computing gradients with epsilon={epsilon}...")
    with tf.GradientTape() as tape:
        # Watch the input tensor
        tape.watch(input_image)
        
        # Get model output (logits)
        logits = model_fn(input_image)
        
        # For targeted attack, maximize the probability of a target class
        # For untargeted attack, maximize the loss of the current prediction
        
        # Convert original prediction to one-hot
        one_hot_target = tf.one_hot(orig_pred, num_classes)
        
        # Compute cross-entropy loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot_target,
                logits=logits
            )
        )
    
    # Get gradients of the loss with respect to the input image
    gradients = tape.gradient(loss, input_image)
    
    # Normalize the gradients (FGSM uses the sign of the gradients)
    signed_gradients = tf.sign(gradients)
    
    # For untargeted attack: maximize loss (add gradient)
    # For targeted attack: minimize loss (subtract gradient)
    sign_direction = -1 if targeted else 1
    
    # Generate adversarial example
    perturbation = sign_direction * epsilon * 255.0 * signed_gradients
    adversarial_image = input_image + perturbation
    
    # Clip to valid range [0, 255]
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 255)
    
    # Convert to uint8
    adversarial_image_np = adversarial_image.numpy().astype(np.uint8)
    if len(adversarial_image_np.shape) == 4:
        adversarial_image_np = adversarial_image_np[0]  # Remove batch dimension
    
    # Get prediction on adversarial example
    print("Getting prediction on adversarial example...")
    adv_pred = deeplab_model.predict(adversarial_image_np)
    
    # Calculate difference metrics
    diff_mask = orig_pred != adv_pred
    diff_percentage = np.sum(diff_mask) / np.prod(diff_mask.shape) * 100
    
    # Get name for saving
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Measure generation time
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")
    
    # Visualize results
    print("Creating visualizations...")
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Adversarial image
    plt.subplot(2, 3, 2)
    plt.imshow(adversarial_image_np)
    plt.title(f"Adversarial Image (ε={epsilon})")
    plt.axis('off')
    
    # Perturbation (amplified for visibility)
    plt.subplot(2, 3, 3)
    perturbation_np = np.abs(image - adversarial_image_np)
    perturbation_amp = np.clip(perturbation_np * 10, 0, 255).astype(np.uint8)
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
    
    plt.suptitle(f"FGSM Results (ε={epsilon}, {'Targeted' if targeted else 'Untargeted'})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{image_name}_fgsm_eps_{epsilon}.png")
    plt.savefig(output_path)
    plt.close()
    
    # Save adversarial image
    adv_path = os.path.join(output_dir, f"{image_name}_adversarial_eps_{epsilon}.png")
    Image.fromarray(adversarial_image_np).save(adv_path)
    
    # Save summary
    summary_path = os.path.join(output_dir, f"{image_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"FGSM Results Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Attack type: {'Targeted' if targeted else 'Untargeted'}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Generation time: {generation_time:.2f} seconds\n")
        f.write(f"  - Pixels changed: {diff_percentage:.2f}%\n")
        f.write(f"  - Max perturbation: {np.max(perturbation_np):.2f}\n")
        f.write(f"  - Mean perturbation: {np.mean(perturbation_np):.2f}\n")
    
    # Clean up
    deeplab_model.close()
    
    print(f"Results saved to {output_dir}")
    print(f"Visualization: {output_path}")
    print(f"Adversarial image: {adv_path}")
    print(f"Summary: {summary_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate adversarial examples using FGSM")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--model_path", type=str, 
                       default="models/deeplabv3_cityscapes_train/frozen_inference_graph.pb",
                       help="Path to frozen model")
    parser.add_argument("--epsilon", type=float, default=0.01,
                       help="Perturbation strength (default: 0.01)")
    parser.add_argument("--targeted", action="store_true",
                       help="Use targeted attack (default: False)")
    parser.add_argument("--output_dir", type=str, default="fixed_fgsm_results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    generate_fgsm_with_eager(
        args.image_path,
        args.model_path,
        args.epsilon,
        args.targeted,
        output_dir=args.output_dir
    ) 