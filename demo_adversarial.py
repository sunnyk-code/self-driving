import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
from FGSM import get_model, preprocess, preprocess_mask, create_adversarial_pattern, create_colormap

def plot_results(original_img, original_mask, original_pred, 
                adversarial_img, adversarial_pred, 
                perturbation, epsilon, save_path):
    """
    Create a figure showing the complete pipeline:
    1. Original image and its segmentation
    2. Adversarial perturbation
    3. Adversarial image and its segmentation
    """
    plt.figure(figsize=(20, 10))
    
    # Original image and its segmentation
    plt.subplot(231)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(232)
    plt.title('Ground Truth Segmentation')
    plt.imshow(original_mask)
    plt.axis('off')
    
    plt.subplot(233)
    plt.title('Original Prediction')
    plt.imshow(original_pred)
    plt.axis('off')
    
    # Adversarial results
    plt.subplot(234)
    plt.title(f'Adversarial Image (ε={epsilon:.3f})')
    plt.imshow(adversarial_img)
    plt.axis('off')
    
    plt.subplot(235)
    plt.title('Perturbation (Magnified)')
    # Normalize perturbation for visualization
    pert_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
    plt.imshow(pert_vis)
    plt.axis('off')
    
    plt.subplot(236)
    plt.title('Adversarial Prediction')
    plt.imshow(adversarial_pred)
    plt.axis('off')
    
    plt.suptitle('Effect of FGSM Attack on Semantic Segmentation', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main(num_samples=5):  # Process 5 images by default
    # Create output directory
    output_dir = 'proposal_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading segmentation model...")
    model = get_model()
    
    # Get list of images
    image_dir = 'data/cityscapes/train/img'
    all_images = glob.glob(os.path.join(image_dir, '*.png'))
    if not all_images:
        print("No images found! Please check the image directory.")
        return
        
    # Select sample images
    if len(all_images) > num_samples:
        selected_images = np.random.choice(all_images, num_samples, replace=False)
    else:
        selected_images = all_images
        print(f"Warning: Only found {len(all_images)} images")
    
    epsilons = [0.01, 0.1, 0.15]
    
    for idx, image_path in enumerate(selected_images, 1):
        print(f"\nProcessing image {idx}/{len(selected_images)}: {os.path.basename(image_path)}")
        mask_path = image_path.replace('/img/', '/label/')
        
        if not os.path.exists(mask_path):
            print(f"No corresponding mask found for {image_path}, skipping...")
            continue
        
        try:
            # Load and preprocess image
            print("Loading and preprocessing image and mask...")
            image_raw = tf.io.read_file(image_path)
            image_decoded = tf.image.decode_image(image_raw, channels=3)
            image_decoded.set_shape([None, None, 3])
            image_preprocessed = preprocess(image_decoded)
            print(f"Image preprocessed shape: {image_preprocessed.shape}")
            
            # Load and preprocess mask
            mask_raw = tf.io.read_file(mask_path)
            mask_decoded = tf.image.decode_image(mask_raw, channels=3)  # Changed to 3 channels for RGB masks
            mask_decoded.set_shape([None, None, 3])
            print(f"Loaded mask shape: {mask_decoded.shape}")
            print(f"Unique values in mask before preprocessing: {np.unique(mask_decoded.numpy())}")
            mask_preprocessed = preprocess_mask(mask_decoded)
            print(f"Mask preprocessed shape: {mask_preprocessed.shape}")
            
            # Verify shapes match
            if mask_preprocessed.shape[1:3] != image_preprocessed.shape[1:3]:
                raise ValueError(f"Shape mismatch: mask {mask_preprocessed.shape} vs image {image_preprocessed.shape}")
            
            # Get original prediction
            print("Getting original prediction...")
            original_pred = model(image_preprocessed)
            print(f"Original prediction shape: {original_pred.shape}")
            
            # Create adversarial pattern
            print("Creating adversarial pattern...")
            perturbations = create_adversarial_pattern(image_preprocessed, mask_preprocessed)
            print(f"Perturbation shape: {perturbations.shape}")
            
            for eps in epsilons:
                print(f"Processing epsilon = {eps}")
                # Create adversarial image
                adv_image = image_preprocessed + eps * perturbations
                adv_image = tf.clip_by_value(adv_image, -1, 1)
                print(f"Adversarial image shape: {adv_image.shape}")
                
                # Get prediction on adversarial image
                adv_pred = model(adv_image)
                print(f"Adversarial prediction shape: {adv_pred.shape}")
                
                # Prepare images for visualization
                original_img = ((image_preprocessed[0].numpy() + 1) * 127.5).astype(np.uint8)
                adv_img = ((adv_image[0].numpy() + 1) * 127.5).astype(np.uint8)
                pert = perturbations[0].numpy()
                
                # Convert predictions to colored segmentation maps
                original_pred_vis = create_colormap(tf.argmax(original_pred[0], axis=-1).numpy())
                adv_pred_vis = create_colormap(tf.argmax(adv_pred[0], axis=-1).numpy())
                mask_vis = create_colormap(tf.argmax(mask_preprocessed[0], axis=-1).numpy())
                
                # Plot and save results
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(output_dir, f'{base_name}_eps{eps:.3f}.png')
                plot_results(
                    original_img, mask_vis, original_pred_vis,
                    adv_img, adv_pred_vis, pert, eps, save_path
                )
                print(f"Saved visualization to {save_path}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate adversarial examples for semantic segmentation')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of images to process')
    args = parser.parse_args()
    
    main(args.num_samples)
    print("\nDemo completed! Check the 'proposal_results' directory for visualizations.") 