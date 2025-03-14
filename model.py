import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import cv2  # Import OpenCV for color detection
import numpy as np  # Import NumPy for numerical operations

# Load ResNet model with updated weights syntax
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Mapping predicted class numbers to actual car models
car_model_mapping = {
    436: "Toyota Corolla",
    545: "Honda Civic",
    231: "Ford Mustang",
    878: "Chevrolet Malibu",
    # Add more mappings based on your dataset
}

# FUNCTION: Predict Car Model
def predict_car_model(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = output.max(1)

    # Look up the car model name based on predicted class
    predicted_class_name = car_model_mapping.get(predicted_class.item(), "Unknown Car Model")
    
    return f"Predicted car model: {predicted_class_name}"

# FUNCTION: Detect Car Color
def detect_car_color(image_path):
    # Load the image with OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB color space

    # Resize image to reduce computation time
    image = cv2.resize(image, (200, 200))

    # Convert image to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculate the average hue of the image
    color = np.mean(hsv, axis=(0, 1))  # Mean color across the image

    # Determine the dominant color based on hue
    if 0 < color[0] < 10:
        return "Red"
    elif 100 < color[0] < 140:
        return "Blue"
    elif 30 < color[0] < 60:
        return "Green"
    else:
        return "Other Color"

# Set dataset path
data_path = r"D:\Users\Harshith\car_project\Car-Similarity-Search-main\cars_imgs"

# Get all image files from dataset
image_files = glob.glob(os.path.join(data_path, "*.jpg"))

# Test with the first image
if image_files:
    test_image = image_files[0]  # Pick the first image from the dataset
    print(f"Testing on image: {test_image}")
    
    car_model = predict_car_model(test_image)
    car_color = detect_car_color(test_image)
    
    print(car_model)
    print(f"Detected car color: {car_color}")
else:
    print("No images found in dataset folder.")
