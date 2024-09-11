import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SkinLesionMaskGeneratorDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir

        # Ensure image directory exists
        if not os.path.exists(self.image_dir):
            raise ValueError(f"The directory {self.image_dir} does not exist.")

        # List of image files (filtering only .jpg and .png)
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        if len(self.images) == 0:
            raise ValueError(f"No .jpg or .png images found in the directory {self.image_dir}.")

        # Set up default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize to fixed size
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load the image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Remove hair from the image
        #image_no_hair = self.remove_hair(image_np)
        
        # Convert back to PIL Image for further transformations
        #image_no_hair_pil = Image.fromarray(image_no_hair)

        image_pil = Image.fromarray(image_np)
        
        # Apply transformations
        if self.transform:
            image_no_hair_tensor = self.transform(image_np)
        else:
            raise RuntimeError("Transform function is not set.")
        
        return image_no_hair_tensor, img_name

    def remove_hair(self, image):
        try:
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
        except Exception as e:
            raise RuntimeError(f"Error in hair removal process: {e}")
