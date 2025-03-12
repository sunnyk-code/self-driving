import torch
import os
import argparse
from FGSM import cityscapesLoader, process_image, save_visualization
from R2U_Net import R2U_Net  # We'll need to create this file next

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset
    dataset = cityscapesLoader(
        root=args.data_dir,
        split='train',
        is_transform=True,
        img_size=(512, 1024)
    )
    
    # Load model
    model = R2U_Net(img_ch=3, output_ch=19).to(device)
    if os.path.exists(args.weights):
        print("Loading pre-trained weights...")
        model.load_state_dict(torch.load(args.weights))
    else:
        print("Warning: No pre-trained weights found. Model will use random initialization.")
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process specified number of images
    image_paths = dataset.files['train'][:args.num_samples]
    epsilons = [0.01, 0.1, 0.15]
    
    for img_path in image_paths:
        print(f"\nProcessing {img_path}")
        
        # Get corresponding label path
        lbl_path = os.path.join(
            dataset.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png"
        )
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        for eps in epsilons:
            print(f"Generating adversarial example with ε = {eps}")
            results = process_image(model, img_path, lbl_path, eps, device, dataset)
            save_visualization(results, args.output_dir, f"{base_name}_eps{eps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/cityscapes',
                        help='Path to Cityscapes dataset')
    parser.add_argument('--weights', type=str, default='r2unet_cityscapes.pth',
                        help='Path to pre-trained weights')
    parser.add_argument('--output_dir', type=str, default='adversarial_results',
                        help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of images to process')
    args = parser.parse_args()
    main(args) 