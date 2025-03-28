import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Add DeepLabV3Plus-Pytorch repo to path - adjust this to your actual location
sys.path.append('./DeepLabV3Plus-Pytorch')

# Import from the DeepLabV3Plus-Pytorch repo
from network.modeling import deeplabv3plus_mobilenet, deeplabv3plus_resnet101, deeplabv3_mobilenet, deeplabv3_resnet101

class DeepLabV3PlusModel:
    def __init__(self, model_name='deeplabv3plus_mobilenet', checkpoint_path=None):
        """
        Initialize a DeepLabV3+ model pretrained on Cityscapes.
        
        Args:
            model_name: Name of the model architecture to use
            checkpoint_path: Path to checkpoint file (if None, will need to be provided later)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model based on architecture
        if model_name == 'deeplabv3plus_mobilenet':
            self.model = deeplabv3plus_mobilenet(num_classes=19, output_stride=16)
        elif model_name == 'deeplabv3plus_resnet101':
            self.model = deeplabv3plus_resnet101(num_classes=19, output_stride=16)
        elif model_name == 'deeplabv3_mobilenet':
            self.model = deeplabv3_mobilenet(num_classes=19, output_stride=16)
        elif model_name == 'deeplabv3_resnet101':
            self.model = deeplabv3_resnet101(num_classes=19, output_stride=16)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set up transforms for preprocessing
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Add safe_load option to handle PyTorch 2.6 security changes
        try:
            # First try with weights_only=True (new default in PyTorch 2.6)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            print(f"First loading attempt failed: {e}")
            print("Trying with weights_only=False (less secure but compatible with older checkpoints)")
            # If that fails, try with weights_only=False (less secure but compatible with older models)
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as nested_e:
                # If that still fails, try with pickle module directly
                print(f"Second loading attempt failed: {nested_e}")
                print("Trying with torch serialization safe globals context...")
                
                # Try with safe_globals context manager if available (PyTorch 2.6+)
                try:
                    from torch.serialization import safe_globals
                    import numpy as np
                    with safe_globals(["numpy.core.multiarray.scalar"]):
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                except ImportError:
                    # For older PyTorch versions without safe_globals
                    raise RuntimeError("Could not load checkpoint. Your PyTorch version might be incompatible with this checkpoint format.")
        
        # Extract model state from checkpoint
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully")

    def preprocess(self, image):
        """
        Preprocess a numpy image to PyTorch tensor.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            PyTorch tensor (1, 3, H, W)
        """
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transform
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    def inverse_transform(self, tensor):
        """
        Convert a normalized tensor back to image space.
        
        Args:
            tensor: Normalized PyTorch tensor [3, H, W]
            
        Returns:
            Un-normalized PyTorch tensor [3, H, W] with values in [0, 1]
        """
        # First, un-normalize using the ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        
        # Undo the normalization: x = (normalized * std) + mean
        return tensor * std + mean
    
    def predict(self, image):
        """
        Run prediction on an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Segmentation mask as numpy array (H, W)
        """
        with torch.no_grad():
            input_tensor = self.preprocess(image)
            
            # Handle different output formats
            try:
                # Try accessing as dictionary first (common in DeepLabV3+ implementations)
                output = self.model(input_tensor)
                if isinstance(output, dict) and 'out' in output:
                    output = output['out']
                elif isinstance(output, dict) and len(output) > 0:
                    # Try getting the first value if it's a dict but doesn't have 'out'
                    output = next(iter(output.values()))
                # If output is already a tensor, use it directly
                
                # Get class predictions
                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                return pred
                
            except Exception as e:
                print(f"Error during prediction: {e}")
                print("Trying alternative output handling...")
                
                # Alternative approach for models with different output structure
                output = self.model(input_tensor)
                if isinstance(output, torch.Tensor):
                    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                    return pred
                else:
                    raise TypeError(f"Unexpected model output type: {type(output)}. Expected dict with 'out' key or tensor.")
    
    def generate_adversarial_fgsm(self, image, epsilon=0.01, targeted=False):
        """
        Generate adversarial example using FGSM.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            epsilon: Perturbation magnitude
            targeted: Whether to perform targeted attack
            
        Returns:
            Tuple of (adversarial_image, perturbation)
        """
        # Convert image to tensor with gradients
        input_tensor = self.preprocess(image)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, dict) and 'out' in output:
            output = output['out']
        elif isinstance(output, dict) and len(output) > 0:
            # Try getting the first value if it's a dict but doesn't have 'out'
            output = next(iter(output.values()))
        # If output is already a tensor, use it directly
        
        # Get original prediction
        pred_class = torch.argmax(output, dim=1)  # Shape: (1, H, W)
        
        # Create loss based on prediction
        criterion = torch.nn.CrossEntropyLoss()
        
        # For untargeted attack: maximize loss of current prediction
        if targeted:
            # For targeted attack, we'd use a specific target class (cycling through classes)
            target_class = (pred_class + 1) % 19  # Cityscapes has 19 classes
            loss = criterion(output, target_class)
        else:
            # For untargeted attack, we use current prediction
            loss = criterion(output, pred_class)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Get sign of gradients
        grad_sign = input_tensor.grad.sign()
        
        # Apply perturbation in the opposite direction for untargeted attack
        perturbation_direction = -1 if not targeted else 1
        perturbed_tensor = input_tensor + (perturbation_direction * epsilon * grad_sign)
        
        # Convert back to image space
        with torch.no_grad():
            perturbed_tensor = perturbed_tensor.detach()
            perturbed_tensor = self.inverse_transform(perturbed_tensor.squeeze(0))
            perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)
            perturbed_image = (perturbed_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Calculate perturbation in image space
        perturbation = perturbed_image.astype(np.float32) - image.astype(np.float32)
        
        return perturbed_image, perturbation
    
    def close(self):
        """Clean up resources if needed."""
        pass  # PyTorch handles this automatically