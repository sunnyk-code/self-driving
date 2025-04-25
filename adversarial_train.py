import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image

from deeplab_cityscapes_pretrained import DeepLabV3PlusModel
from cityscapes_utils import get_dataset_files, cityscapes_classes

# Dataset for Cityscapes images and labels
class CityscapesSegmentationDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx])
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(np.array(label)).long()
        # Remap invalid labels to 255 (ignore index)
        label[~((label >= 0) & (label <= 18))] = 255
        return img, label

def fgsm_attack(images, labels, model, criterion, epsilon):
    # Detach and clone to avoid modifying the original tensor
    images_adv = images.detach().clone()
    images_adv.requires_grad = True
    outputs = model.model(images_adv)
    if isinstance(outputs, dict) and 'out' in outputs:
        outputs = outputs['out']
    loss = criterion(outputs, labels)
    model.model.zero_grad()
    loss.backward()
    data_grad = images_adv.grad.data
    perturbed_images = images_adv + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images.detach()

def compute_metrics(preds, labels, num_classes=19, ignore_index=255):
    """
    Compute mean IoU and pixel accuracy for a batch.
    preds, labels: (N, H, W) numpy arrays or torch tensors
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    ious = []
    total_correct = 0
    total_labeled = 0
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        valid = (labels != ignore_index)
        intersection = np.logical_and(pred_mask, label_mask) & valid
        union = np.logical_or(pred_mask, label_mask) & valid
        inter = np.sum(intersection)
        uni = np.sum(union)
        if uni > 0:
            ious.append(inter / uni)
    # Pixel accuracy
    valid = (labels != ignore_index)
    total_correct = np.sum((preds == labels) & valid)
    total_labeled = np.sum(valid)
    pix_acc = total_correct / (total_labeled + 1e-10)
    mean_iou = np.mean(ious) if ious else 0.0
    return mean_iou, pix_acc

def adversarial_train(model, train_loader, device, epochs=20, epsilon=0.03, lr=1e-4, save_path=None, test_mode=False):
    import matplotlib.pyplot as plt
    model.model.train()
    optimizer = optim.Adam(model.model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Estimate training time
    import time
    num_batches = len(train_loader)
    if num_batches > 0:
        print("Estimating average batch time...")
        batch_times = []
        train_iter = iter(train_loader)
        for i in range(min(3, num_batches)):
            images, labels = next(train_iter)
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model.model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            loss = criterion(outputs, labels)
            adv_images = fgsm_attack(images, labels, model, criterion, epsilon)
            adv_outputs = model.model(adv_images)
            if isinstance(adv_outputs, dict) and 'out' in adv_outputs:
                adv_outputs = adv_outputs['out']
            adv_loss = criterion(adv_outputs, labels)
            total_loss = (loss + adv_loss) / 2
            total_loss.backward()
            optimizer.step()
            batch_times.append(time.time() - start_time)
        avg_batch_time = sum(batch_times) / len(batch_times)
        est_total_time = avg_batch_time * num_batches * epochs
        est_minutes = est_total_time / 60
        est_hours = est_minutes / 60
        print(f"Estimated average batch time: {avg_batch_time:.2f} seconds")
        print(f"Estimated total training time: {est_total_time:.0f} seconds (~{est_minutes:.1f} min, ~{est_hours:.2f} hours)\n")

    print()

    if test_mode:
        print("[TEST MODE] Running on a small subset of the data and for 2 epochs only.")
        epochs = 2

    print(f"Starting adversarial training for {epochs} epochs...")

    print()

    clean_losses = []
    adv_losses = []
    total_losses = []
    epoch_clean_losses = []
    epoch_adv_losses = []
    epoch_total_losses = []
    # Open file to log batch losses
    with open('batch_losses.txt', 'w') as log_file:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs} started.")
            model.model.train()
            running_loss = 0.0
            running_miou = 0.0
            running_pixacc = 0.0
            epoch_clean_loss = 0.0
            epoch_adv_loss = 0.0
            epoch_total_loss = 0.0
            n_batches = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"  Batch {batch_idx+1} started.")
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # Standard training step
                outputs = model.model(images)
                if isinstance(outputs, dict) and 'out' in outputs:
                    outputs = outputs['out']
                loss = criterion(outputs, labels)
                print(f"    Clean loss: {loss.item():.4f}")

                # Adversarial training step
                adv_images = fgsm_attack(images, labels, model, criterion, epsilon)
                adv_outputs = model.model(adv_images)
                if isinstance(adv_outputs, dict) and 'out' in adv_outputs:
                    adv_outputs = adv_outputs['out']
                adv_loss = criterion(adv_outputs, labels)
                print(f"    Adversarial loss: {adv_loss.item():.4f}")

                total_loss = (loss + adv_loss) / 2

                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

                # Track losses
                clean_losses.append(loss.item())
                adv_losses.append(adv_loss.item())
                total_losses.append(total_loss.item())
                epoch_clean_loss += loss.item()
                epoch_adv_loss += adv_loss.item()
                epoch_total_loss += total_loss.item()

                # Metrics (on clean images)
                preds = torch.argmax(outputs, dim=1)
                miou, pixacc = compute_metrics(preds, labels)
                running_miou += miou
                running_pixacc += pixacc
                n_batches += 1

                # Log to file
                log_file.write(f"Epoch {epoch+1}, Batch {batch_idx+1}, Clean Loss: {loss.item():.4f}, Adversarial Loss: {adv_loss.item():.4f}, Total Loss: {total_loss.item():.4f}, mIoU: {miou:.4f}, PixAcc: {pixacc:.4f}\n")
                log_file.flush()

                print(f"    Batch {batch_idx+1} finished. Total loss: {total_loss.item():.4f}, mIoU: {miou:.4f}, PixAcc: {pixacc:.4f}")

            avg_loss = running_loss / n_batches
            avg_miou = running_miou / n_batches
            avg_pixacc = running_pixacc / n_batches
            epoch_clean_losses.append(epoch_clean_loss / n_batches)
            epoch_adv_losses.append(epoch_adv_loss / n_batches)
            epoch_total_losses.append(epoch_total_loss / n_batches)
            print(f"Epoch {epoch+1}/{epochs} finished. Loss: {avg_loss:.4f}, Mean IoU: {avg_miou:.4f}, Pixel Acc: {avg_pixacc:.4f}")

            # Save checkpoint after each epoch if save_path is provided
            if save_path:
                torch.save({'model_state': model.model.state_dict()}, save_path)
                print(f"Checkpoint saved to {save_path}")
    print("Adversarial training completed.")

    # Plot and save loss curves by epoch
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(epoch_clean_losses)+1), epoch_clean_losses, label='Clean Loss')
    plt.plot(range(1, len(epoch_adv_losses)+1), epoch_adv_losses, label='Adversarial Loss')
    plt.plot(range(1, len(epoch_total_losses)+1), epoch_total_losses, label='Total Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves (per Epoch)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curves.png')
    print('Loss curves saved to loss_curves.png')

def evaluate(model, data_loader, device, results_file='test_results.txt'):
    model.model.eval()
    total_miou = 0.0
    total_pixacc = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            preds = torch.argmax(outputs, dim=1)
            miou, pixacc = compute_metrics(preds, labels)
            total_miou += miou
            total_pixacc += pixacc
            n_batches += 1
    avg_miou = total_miou / n_batches
    avg_pixacc = total_pixacc / n_batches
    result_str = f"Test set results: Mean IoU: {avg_miou:.4f}, Pixel Acc: {avg_pixacc:.4f}\n"
    print(result_str)
    with open(results_file, 'w') as f:
        f.write(result_str)

if __name__ == "__main__":
    # Set paths
    DATA_ROOT = './'  # Adjust if needed
    IMG_ROOT = './leftImg8bit_trainvaltest'
    LABEL_ROOT = './gtFine_trainvaltest'
    CHECKPOINT_PATH = './checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
    SAVE_PATH = './checkpoints/adv_trained_deeplabv3plus_mobilenet_cityscapes_os16.pth'

    # Get train set file lists
    image_paths, label_paths = get_dataset_files(DATA_ROOT, split='train')
    print(f"Found {len(image_paths)} training images.")

    # Use the same preprocessing as in DeepLabV3PlusModel
    model = DeepLabV3PlusModel(model_name='deeplabv3plus_mobilenet', checkpoint_path=CHECKPOINT_PATH)
    transform = model.transform

    # DataLoader
    dataset = CityscapesSegmentationDataset(image_paths, label_paths, transform=transform)
    # TEST MODE: Use a small subset for quick testing
    test_mode = False  # Set to False to run on full dataset
    if test_mode:
        small_subset = Subset(dataset, range(10))  # Use first 10 samples
        train_loader = DataLoader(small_subset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    else:
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    adversarial_train(model, train_loader, device, epochs=20, epsilon=0.1, lr=1e-4, save_path=SAVE_PATH, test_mode=test_mode)

    # Evaluate on test set
    test_image_paths, test_label_paths = get_dataset_files(DATA_ROOT, split='test')
    print(f"Found {len(test_image_paths)} test images.")
    if len(test_image_paths) > 0:
        test_dataset = CityscapesSegmentationDataset(test_image_paths, test_label_paths, transform=transform)
        if test_mode:
            from torch.utils.data import Subset
            test_subset = Subset(test_dataset, range(10))  # Use first 10 test samples
            test_loader = DataLoader(test_subset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)
            print("[TEST MODE] Evaluating on a small subset of the test set.")
        else:
            test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)
        # Load best model if needed
        if os.path.exists(SAVE_PATH):
            checkpoint = torch.load(SAVE_PATH, map_location=device)
            model.model.load_state_dict(checkpoint['model_state'])
        evaluate(model, test_loader, device)
    else:
        print("No test images found.")
