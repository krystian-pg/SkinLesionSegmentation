import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import random

class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.image_to_mask = {img: self.get_mask_name(img) for img in self.images}

        # Ensure transforms are applied, with default resizing to a fixed size
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize to fixed size
                transforms.ToTensor()
            ])
        if self.mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize to fixed size
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = self.image_to_mask[img_name]
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Open the image and convert it to a numpy array
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Remove hair from the image
        #image_no_hair = self.remove_hair(image)
        
        # Convert the processed image back to PIL format for transformations
        image_no_hair = Image.fromarray(image)
        
        # Open the mask
        mask = Image.open(mask_path).convert("L")
        
        # Apply transformations (including resizing and conversion to tensors)
        image_no_hair = self.transform(image_no_hair)
        mask = self.mask_transform(mask)
        
        return image_no_hair, mask

    def get_mask_name(self, img_name):
        if img_name.endswith('.jpg'):
            return img_name.replace('.jpg', '_segmentation.png')
        elif img_name.endswith('.png'):
            return img_name.replace('.png', '_segmentation.png')
        else:
            raise ValueError(f"Unexpected file extension in {img_name}")
        
    def remove_hair(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast using adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Use morphological operations to detect hair
        kernel_size = (9, 9)  # Smaller kernel size for more precise hair detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Use a threshold to create a binary mask of the hair
        _, binary_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
        
        # Dilate the mask to cover hair more effectively
        dilated_mask = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=1)
        
        # Inpaint the original image using the dilated mask
        inpainted_image = cv2.inpaint(image, dilated_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        
        return inpainted_image

    def visualize_transformations(self, num_images=12):
        # Randomly select a subset of images
        selected_indices = random.sample(range(len(self.images)), num_images)
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 2))
        
        for i, idx in enumerate(selected_indices):
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            original_image = np.array(Image.open(img_path).convert("RGB"))
            
            # Get the transformed image with hair removed
            transformed_image, mask = self.__getitem__(idx)
            
            # Convert the transformed image from a PyTorch tensor to a NumPy array if needed
            if torch.is_tensor(transformed_image):
                transformed_image = transformed_image.permute(1, 2, 0).numpy()
            
            # Display the original and transformed images
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f"Original Image {img_name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(transformed_image)
            axes[i, 1].set_title(f"Transformed Image {img_name}")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
