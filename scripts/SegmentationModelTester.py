import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import pandas as pd


class SegmentationModelTester:
    def __init__(self, model, device, test_loader):
        """
        Initializes the tester with a model, device, and test DataLoader.

        :param model: Trained segmentation model
        :param device: Device to run the model on (e.g., 'cpu' or 'cuda')
        :param test_loader: DataLoader for the test dataset
        """
        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader

    def generate_predictions(self):
        """
        Generates predictions for the test dataset and stores the results.

        :return: Tuple of lists containing predicted masks and ground truth masks
        """
        predicted_masks_list = []
        ground_truth_masks_list = []

        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to calculate gradients for inference
            for batch in tqdm(self.test_loader, desc="Generating Masks"):
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Run the model on the input batch
                outputs = self.model(images)

                # Apply sigmoid to get probabilities
                predictions = torch.sigmoid(outputs)

                # Threshold the predictions to get binary masks
                predictions = (predictions > 0.5).float()

                # Move predictions and masks to CPU and convert them to NumPy arrays
                predictions_np = predictions.cpu().numpy()
                masks_np = masks.cpu().numpy()

                # Append to the lists
                predicted_masks_list.extend(predictions_np)
                ground_truth_masks_list.extend(masks_np)

        return predicted_masks_list, ground_truth_masks_list

    def calculate_metrics(self, predicted_masks_list, ground_truth_masks_list, smooth=1e-6):
        """
        Calculates IoU, Dice coefficient, Precision, Recall, and F1-score.

        :param predicted_masks_list: List of predicted masks (NumPy arrays)
        :param ground_truth_masks_list: List of ground truth masks (NumPy arrays)
        :param smooth: Smoothing factor to avoid division by zero
        :return: Dictionary containing mean values for IoU, Dice, Precision, Recall, and F1-score
        """
        iou_list = []
        dice_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        for pred_mask, true_mask in zip(predicted_masks_list, ground_truth_masks_list):
            # Flatten the arrays to simplify metric calculation
            pred_mask = pred_mask.squeeze().flatten()
            true_mask = true_mask.squeeze().flatten()

            # Calculate intersection and union
            intersection = np.sum(pred_mask * true_mask)
            union = np.sum(pred_mask) + np.sum(true_mask) - intersection

            # Calculate IoU
            iou = (intersection + smooth) / (union + smooth)
            iou_list.append(iou)

            # Calculate Dice coefficient
            dice = (2. * intersection + smooth) / (np.sum(pred_mask) + np.sum(true_mask) + smooth)
            dice_list.append(dice)

            # Calculate Precision
            precision = (intersection + smooth) / (np.sum(pred_mask) + smooth)
            precision_list.append(precision)

            # Calculate Recall
            recall = (intersection + smooth) / (np.sum(true_mask) + smooth)
            recall_list.append(recall)

            # Calculate F1-score
            f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)
            f1_list.append(f1)

        return {
            'mean_iou': np.mean(iou_list),
            'mean_dice': np.mean(dice_list),
            'mean_precision': np.mean(precision_list),
            'mean_recall': np.mean(recall_list),
            'mean_f1': np.mean(f1_list)
        }

    def test(self):
        """
        Runs the complete testing pipeline: generates predictions and calculates metrics.

        :return: Dictionary containing mean values for IoU, Dice, Precision, Recall, and F1-score
        """
        predicted_masks_list, ground_truth_masks_list = self.generate_predictions()
        metrics = self.calculate_metrics(predicted_masks_list, ground_truth_masks_list)
        return metrics
