import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from FGSM import cityscapesLoader
from R2U_Net import R2U_Net

# Hyperparameters (same as notebook)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if device.type == 'cpu':
    print("WARNING: Training on CPU will be very slow. Consider using GPU if available.")
learning_rate = 1e-6
train_epochs = 8
n_classes = 19
batch_size = 1
num_workers = 1

def cross_entropy2d(input, target, weight=None, size_average=True):
    """Cross entropy loss adapted from the notebook."""
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

class runningScore(object):
    """Metrics calculation class from notebook."""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        hist = self.confusion_matrix
        
        TP = np.diag(hist)
        TN = hist.sum() - hist.sum(axis=1) - hist.sum(axis=0) + np.diag(hist)
        FP = hist.sum(axis=1) - TP
        FN = hist.sum(axis=0) - TP
        
        specif_cls = (TN) / (TN + FP + 1e-6)
        specif = np.nanmean(specif_cls)
        
        sensti_cls = (TP) / (TP + FN + 1e-6)
        sensti = np.nanmean(sensti_cls)
        
        prec_cls = (TP) / (TP + FP + 1e-6)
        prec = np.nanmean(prec_cls)
        
        f1 = (2 * prec * sensti) / (prec + sensti + 1e-6)
        
        return {
            "Specificity": specif,
            "Senstivity": sensti,
            "F1": f1,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def get_metrics(gt_label, pred_label):
    """Additional metrics from notebook."""
    acc = skm.accuracy_score(gt_label, pred_label, normalize=True)
    js = skm.jaccard_score(gt_label, pred_label, average='micro')
    return [acc, js]

def train(train_loader, model, optimizer, epoch_i, epoch_total):
    """Training function from notebook."""
    count = 0
    loss_list = []
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch_i + 1}')
    for i, (images, labels) in enumerate(progress_bar):
        count += 1
        model.train()

        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = cross_entropy2d(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Print batch statistics every 50 batches
        if i % 50 == 0:
            print(f'\nBatch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        loss_list.append(loss)
        
    return loss_list

def validate(val_loader, model, epoch_i):
    """Validation function from notebook."""
    model.eval()
    running_metrics_val = runningScore(19)
    
    acc_sh = []
    js_sh = []
    
    with torch.no_grad():
        for image_num, (val_images, val_labels) in tqdm(enumerate(val_loader)):
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            
            val_pred = model(val_images)
            pred = val_pred.data.max(1)[1].cpu().numpy()
            gt = val_labels.data.cpu().numpy()
            
            running_metrics_val.update(gt, pred)
            sh_metrics = get_metrics(gt.flatten(), pred.flatten())
            acc_sh.append(sh_metrics[0])
            js_sh.append(sh_metrics[1])

    score = running_metrics_val.get_scores()
    running_metrics_val.reset()
    
    acc_s = sum(acc_sh)/len(acc_sh)
    js_s = sum(js_sh)/len(js_sh)
    score["acc"] = acc_s
    score["js"] = js_s
    
    print("Different Metrics were: ", score)  
    return score

def main():
    # Initialize dataset and loaders
    train_data = cityscapesLoader(
        root='data/cityscapes',
        split='train'
    )
    
    val_data = cityscapesLoader(
        root='data/cityscapes',
        split='val'
    )
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize model, optimizer
    model = R2U_Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store metrics
    loss_all_epochs = []
    Specificity_ = []
    Senstivity_ = []
    F1_ = []
    acc_ = []
    js_ = []
    
    # Training loop
    for epoch_i in range(train_epochs):
        print(f"\nEpoch {epoch_i + 1}")
        print("-------------------------------")
        t1 = time.time()
        loss_i = train(train_loader, model, optimizer, epoch_i, train_epochs)
        loss_all_epochs.append(loss_i)
        t2 = time.time()
        print("It took: ", t2-t1, " unit time")
        
        # Validation
        score = validate(val_loader, model, epoch_i)
        Specificity_.append(score["Specificity"])
        Senstivity_.append(score["Senstivity"])
        F1_.append(score["F1"])
        acc_.append(score["acc"])
        js_.append(score["js"])
        
        # Save model after each epoch
        torch.save(model.state_dict(), f'r2unet_cityscapes_epoch{epoch_i+1}.pth')
    
    # Plot training loss
    loss_1d_list = [item for sublist in loss_all_epochs for item in sublist]
    loss_list_numpy = [loss.cpu().detach().numpy() for loss in loss_1d_list]
    plt.figure()
    plt.xlabel("Images used in training epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.plot(loss_list_numpy)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Plot metrics
    plt.figure()
    x = range(1, train_epochs + 1)
    plt.plot(x, Specificity_, label='Specificity')
    plt.plot(x, Senstivity_, label='Senstivity')
    plt.plot(x, F1_, label='F1 Score')
    plt.plot(x, acc_, label='Accuracy')
    plt.plot(x, js_, label='Jaccard Score')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig('metrics.png')
    plt.close()

if __name__ == "__main__":
    main() 