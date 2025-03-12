import torch
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

NUM_CLASSES = 19

class cityscapesLoader(data.Dataset):
    colors = [  
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, root, split="train", is_transform=True, img_size=(512, 1024)):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 19
        self.img_size = img_size
        self.files = {}
        
        possible_gtFine_paths = [
            os.path.join(self.root, "gtFine", self.split),  
            os.path.join(self.root, "gtFine_trainvaltest", "gtFine", self.split),  
            os.path.join(self.root, "cityscapes", "gtFine_trainvaltest", "gtFine", self.split), 
            os.path.join(self.root, self.split) 
        ]
        
        possible_leftImg_paths = [
            os.path.join(self.root, "leftImg8bit", self.split),  
            os.path.join(self.root, "leftImg8bit_trainvaltest", "leftImg8bit", self.split),  
            os.path.join(self.root, "cityscapes", "leftImg8bit_trainvaltest", "leftImg8bit", self.split),  
            os.path.join(self.root, "leftImg8bit_" + self.split)  
        ]
        
        self.annotations_base = None
        for path in possible_gtFine_paths:
            if os.path.exists(path):
                self.annotations_base = path
                print(f"Found gtFine data in: {path}")
                break
        
        if self.annotations_base is None:
            raise Exception(f"Could not find gtFine data in any of: {possible_gtFine_paths}")
            
        self.images_base = None
        for path in possible_leftImg_paths:
            if os.path.exists(path):
                self.images_base = path
                print(f"Found leftImg8bit data in: {path}")
                break
        
        if self.images_base:
            self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix=".png")
            print(f"Found {len(self.files[split])} images in leftImg8bit")
        else:
            print("Warning: leftImg8bit data not found. Only ground truth data will be available.")
            self.files[split] = [
                f for f in self.recursive_glob(rootdir=self.annotations_base, suffix=".png")
                if "labelIds" in f
            ]
            print(f"Found {len(self.files[split])} label files in gtFine")
        
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = [
            "unlabelled", "road", "sidewalk", "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation", "terrain", "sky", "person",
            "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
        ]
        
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def recursive_glob(self, rootdir=".", suffix=""):
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        """Get image and label pair."""
        try:
            if self.images_base:
                # If we have image data, use it
                img_path = self.files[self.split][index].rstrip()
                lbl_path = os.path.join(
                    self.annotations_base,
                    img_path.split(os.sep)[-2],
                    os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
                )
                
                if not os.path.exists(lbl_path):
                    raise FileNotFoundError(f"Label file not found: {lbl_path}")
                
                img = Image.open(img_path)
                img = np.array(img, dtype=np.uint8)
            else:
                lbl_path = self.files[self.split][index].rstrip()
                print(f"Warning: Using dummy image for {lbl_path}")
                img = np.zeros((1024, 2048, 3), dtype=np.uint8)  # Default Cityscapes size
            
            lbl = Image.open(lbl_path)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

            if self.is_transform:
                img, lbl = self.transform(img, lbl)

            return img, lbl
            
        except Exception as e:
            print(f"Error loading file {self.files[self.split][index]}: {str(e)}")
            dummy_img = torch.zeros((3, self.img_size[0], self.img_size[1]))
            dummy_lbl = torch.ones((self.img_size[0], self.img_size[1])) * self.ignore_index
            return dummy_img, dummy_lbl

    def transform(self, img, lbl):
        # Resize image
        img = Image.fromarray(img)
        img = img.resize(self.img_size[::-1], Image.BILINEAR)  # PIL uses (width, height)
        img = np.array(img)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)

        # Resize label
        classes = np.unique(lbl)
        lbl = Image.fromarray(lbl)
        lbl = lbl.resize(self.img_size[::-1], Image.NEAREST)
        lbl = np.array(lbl, dtype=np.int64)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

def create_adversarial_pattern(model, image, target, epsilon, device):
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    
    # Calculate loss
    loss = F.cross_entropy(output, target, ignore_index=250)
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    data_grad = image.grad.data
    
    # Create adversarial example
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    
    return perturbed_image

def process_image(model, image_path, label_path, epsilon, device, dataset):
    """Process a single image and generate adversarial example."""
    img = Image.open(image_path)
    img = np.array(img, dtype=np.uint8)
    
    lbl = Image.open(label_path)
    lbl = dataset.encode_segmap(np.array(lbl, dtype=np.uint8))
    
    img, lbl = dataset.transform(img, lbl)
    
    img = img.unsqueeze(0).to(device)
    lbl = lbl.unsqueeze(0).to(device)
    
    perturbed_image = create_adversarial_pattern(model, img, lbl, epsilon, device)
    
    with torch.no_grad():
        original_pred = model(img)
        adversarial_pred = model(perturbed_image)
    
    original_pred = torch.argmax(original_pred, dim=1)[0].cpu().numpy()
    adversarial_pred = torch.argmax(adversarial_pred, dim=1)[0].cpu().numpy()
    
    return {
        'original_image': img[0].cpu().numpy().transpose(1, 2, 0),
        'perturbed_image': perturbed_image[0].cpu().detach().numpy().transpose(1, 2, 0),
        'original_pred': dataset.decode_segmap(original_pred),
        'adversarial_pred': dataset.decode_segmap(adversarial_pred),
        'ground_truth': dataset.decode_segmap(lbl[0].cpu().numpy())
    }

def save_visualization(results, output_path, base_name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    

    axes[0,0].imshow(results['original_image'])
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    

    axes[0,1].imshow(results['ground_truth'])
    axes[0,1].set_title('Ground Truth')
    axes[0,1].axis('off')
    
    # Original 
    axes[0,2].imshow(results['original_pred'])
    axes[0,2].set_title('Original Prediction')
    axes[0,2].axis('off')
    
    # Adversarial
    axes[1,0].imshow(results['perturbed_image'])
    axes[1,0].set_title('Adversarial Image')
    axes[1,0].axis('off')
    
    #  visualization
    perturbation = results['perturbed_image'] - results['original_image']
    perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
    axes[1,1].imshow(perturbation)
    axes[1,1].set_title('Perturbation')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(results['adversarial_pred'])
    axes[1,2].set_title('Adversarial Prediction')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{base_name}_results.png'))
    plt.close()
