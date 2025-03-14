import os
import glob

data_path = data_path = r"D:\Users\Harshith\car_project\Car-Similarity-Search-main\cars_imgs" # Update with your extracted dataset path

# Check available files
image_files = glob.glob(os.path.join(data_path, "**/*.jpg"), recursive=True)
print("Total images found:", len(image_files))

# Display first few image paths
print(image_files[:5])  
