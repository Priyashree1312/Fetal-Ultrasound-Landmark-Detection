import cv2
import os

IMG_DIR = r"C:\Priyashree_Panda_Research\Dataset\Part_B\images"
MASK_DIR = r"C:\Priyashree_Panda_Research\Dataset\Part_B\masks"

images = os.listdir(IMG_DIR)
print(f"Total Images found: {len(images)}")

for img_name in images[:5]: # Check first 5
    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    # Note: Masks often have slightly different names, like '001_HC_Annotation.png'
    # Adjust the line below to match your mask naming convention
    mask_name = img_name.replace(".png", "_Annotation.png") 
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name))
    
    if mask is None:
        print(f"❌ Error: Mask not found for {img_name}")
    else:
        print(f"✅ Success: {img_name} matches {mask_name} | Sizes: {img.shape} vs {mask.shape}")