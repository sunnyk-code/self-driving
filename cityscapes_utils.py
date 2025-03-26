import os
import glob
import numpy as np
from PIL import Image

# Cityscapes label colors (RGB format)
CITYSCAPES_COLORS = [
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [70, 70, 70],      # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170, 30],    # traffic light
    [220, 220, 0],     # traffic sign
    [107, 142, 35],    # vegetation
    [152, 251, 152],   # terrain
    [70, 130, 180],    # sky
    [220, 20, 60],     # person
    [255, 0, 0],       # rider
    [0, 0, 142],       # car
    [0, 0, 70],        # truck
    [0, 60, 100],      # bus
    [0, 80, 100],      # train
    [0, 0, 230],       # motorcycle
    [119, 11, 32],     # bicycle
]

def create_colormap(segmentation, colormap=CITYSCAPES_COLORS):
    """
    Convert a segmentation mask to an RGB colormap using the provided colors.
    
    Args:
        segmentation: Segmentation mask with integer class IDs
        colormap: List of RGB colors for each class
        
    Returns:
        RGB image where each pixel is colored according to its class
    """
    # Create empty RGB image
    height, width = segmentation.shape[:2]
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign colors based on class IDs
    for class_id, color in enumerate(colormap):
        mask = (segmentation == class_id)
        colored_mask[mask] = color
        
    return colored_mask

def find_label_path(image_path):
    """
    Find the corresponding label path for an image path.
    Handles different Cityscapes directory structures.
    
    Args:
        image_path: Path to an image file
        
    Returns:
        Path to the corresponding label file, or None if not found
    """
    # Common patterns to check
    patterns = [
        # Standard Cityscapes structure
        lambda img: img.replace('leftImg8bit', 'gtFine').replace('_leftImg8bit.png', '_gtFine_labelIds.png'),
        # Simplified structure
        lambda img: img.replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png'),
        # Same directory structure
        lambda img: os.path.join(os.path.dirname(img), os.path.basename(img).replace('_leftImg8bit.png', '_gtFine_labelIds.png')),
    ]
    
    for pattern in patterns:
        label_path = pattern(image_path)
        if os.path.exists(label_path):
            return label_path
    
    return None

def get_dataset_files(root_dir, split='val', city=None):
    """
    Get lists of image and label files from a Cityscapes dataset directory.
    
    Args:
        root_dir: Root directory of the Cityscapes dataset
        split: Data split to use ('train', 'val', or 'test')
        city: Specific city to use (optional)
        
    Returns:
        Tuple of (image_paths, label_paths)
    """
    # Try different directory structures
    possible_img_dirs = [
        os.path.join(root_dir, 'leftImg8bit', split),
        os.path.join(root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', split),
        os.path.join(root_dir, split, 'leftImg8bit'),
    ]
    
    # Find the first valid directory
    img_dir = None
    for dir_path in possible_img_dirs:
        if os.path.isdir(dir_path):
            img_dir = dir_path
            break
    
    # If no valid directory found
    if img_dir is None:
        print(f"Could not find image directory for split {split} in {root_dir}")
        return [], []
    
    # Get image files
    if city:
        # Specific city
        city_dir = os.path.join(img_dir, city)
        if os.path.isdir(city_dir):
            image_paths = sorted(glob.glob(os.path.join(city_dir, '*_leftImg8bit.png')))
        else:
            image_paths = sorted(glob.glob(os.path.join(img_dir, f'*{city}*_leftImg8bit.png')))
    else:
        # All cities
        image_paths = []
        # First try with city subdirectories
        city_dirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        if city_dirs:
            for city_name in city_dirs:
                city_images = sorted(glob.glob(os.path.join(img_dir, city_name, '*_leftImg8bit.png')))
                image_paths.extend(city_images)
        else:
            # No city subdirectories, look directly in img_dir
            image_paths = sorted(glob.glob(os.path.join(img_dir, '*_leftImg8bit.png')))
    
    # Get corresponding label files
    label_paths = []
    for img_path in image_paths:
        label_path = find_label_path(img_path)
        label_paths.append(label_path)
    
    # Filter out images without labels
    if any(label is None for label in label_paths):
        valid_pairs = [(img, label) for img, label in zip(image_paths, label_paths) if label is not None]
        if valid_pairs:
            print(f"Warning: Found {len(image_paths) - len(valid_pairs)} images without corresponding labels")
            image_paths, label_paths = zip(*valid_pairs)
        else:
            print("Warning: No valid image-label pairs found")
            return [], []
    
    return image_paths, label_paths
    
def load_cityscapes_image(image_path):
    """Load a Cityscapes image."""
    try:
        return np.array(Image.open(image_path).convert('RGB'))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_cityscapes_label(label_path):
    """Load a Cityscapes label."""
    try:
        return np.array(Image.open(label_path))
    except Exception as e:
        print(f"Error loading label {label_path}: {e}")
        return None 