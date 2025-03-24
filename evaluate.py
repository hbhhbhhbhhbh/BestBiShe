import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    total_dice_score = 0
    total_pixel_accuracy = 0
    total_iou = 0
    total_f1 = 0
    total_recall = 0
    total_precision = 0
    total_batches = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred= net(image)

            if net.n_classes == 1:
                # For binary classification
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                # Compute Dice score
                total_dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

                # Compute Pixel Accuracy
                pixel_accuracy = (mask_pred == mask_true).float().mean().item()
                total_pixel_accuracy += pixel_accuracy

                # Compute IoU
                intersection = torch.sum(mask_pred * mask_true)
                union = torch.sum(mask_pred) + torch.sum(mask_true)
                iou = intersection / (union - intersection + 1e-6)
                total_iou += iou.item()

                # Compute F1, Recall, and Precision
                true_positives = torch.sum(mask_pred * mask_true)
                false_positives = torch.sum(mask_pred * (1 - mask_true))
                false_negatives = torch.sum((1 - mask_pred) * mask_true)

                precision = true_positives / (true_positives + false_positives + 1e-6)
                recall = true_positives / (true_positives + false_negatives + 1e-6)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

                total_f1 += f1.item()
                total_recall += recall.item()
                total_precision += precision.item()

            else:
                # For multi-class classification
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                mask_pred = F.softmax(mask_pred, dim=1)  # Get probability map
                mask_pred = mask_pred.argmax(dim=1)  # Get predicted class per pixel
                mask_pred = F.one_hot(mask_pred, net.n_classes).permute(0, 3, 1, 2).float()

                # Convert true mask to one-hot encoding
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

                # Compute Dice score
                total_dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # Compute Pixel Accuracy
                pixel_accuracy = (mask_pred.argmax(dim=1) == mask_true.argmax(dim=1)).float().mean().item()
                total_pixel_accuracy += pixel_accuracy

                # Compute IoU for each class
                iou_classwise = []
                f1_classwise = []
                recall_classwise = []
                precision_classwise = []
                for c in range(1, net.n_classes):  # We skip background class (index 0)
                    intersection = torch.sum(mask_pred[:, c] * mask_true[:, c])
                    union = torch.sum(mask_pred[:, c]) + torch.sum(mask_true[:, c])
                    iou = intersection / (union - intersection + 1e-6)
                    iou_classwise.append(iou.item())

                    # Compute F1, Recall, and Precision for each class
                    true_positives = torch.sum(mask_pred[:, c] * mask_true[:, c])
                    false_positives = torch.sum(mask_pred[:, c] * (1 - mask_true[:, c]))
                    false_negatives = torch.sum((1 - mask_pred[:, c]) * mask_true[:, c])

                    precision = true_positives / (true_positives + false_positives + 1e-6)
                    recall = true_positives / (true_positives + false_negatives + 1e-6)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

                    f1_classwise.append(f1.item())
                    recall_classwise.append(recall.item())
                    precision_classwise.append(precision.item())

                mean_iou = sum(iou_classwise) / len(iou_classwise)  # Mean IoU across all classes (ignoring background)
                mean_f1 = sum(f1_classwise) / len(f1_classwise)
                mean_recall = sum(recall_classwise) / len(recall_classwise)
                mean_precision = sum(precision_classwise) / len(precision_classwise)

                total_iou += mean_iou
                total_f1 += mean_f1
                total_recall += mean_recall
                total_precision += mean_precision

            total_batches += 1

    # Calculate average metrics
    avg_dice_score = total_dice_score / total_batches
    avg_pixel_accuracy = total_pixel_accuracy / total_batches
    avg_iou = total_iou / total_batches
    avg_f1 = total_f1 / total_batches
    avg_recall = total_recall / total_batches
    avg_precision = total_precision / total_batches

    net.train()
    return avg_dice_score, avg_pixel_accuracy, avg_iou, avg_f1, avg_recall, avg_precision