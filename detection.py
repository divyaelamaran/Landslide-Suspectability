Python 3.11.7 (main, Dec  7 2023, 09:09:57)  [GCC UCRT 13.2.0 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Import necessary libraries
... import os
... import shutil
... from sklearn.model_selection import train_test_split
... 
... # Paths
... dataset_dir = r'D:/Landslide-Project/dataset'  # Update this to your dataset directory
... image_dir = os.path.join(dataset_dir, 'images')  # Original images
... mask_dir = os.path.join(dataset_dir, 'masks')    # Original masks
... train_img_dir = os.path.join(dataset_dir, 'images', 'train')  # Training images
... val_img_dir = os.path.join(dataset_dir, 'images', 'val')      # Validation images
... test_img_dir = os.path.join(dataset_dir, 'images', 'test')    # Test images
... train_mask_dir = os.path.join(dataset_dir, 'masks', 'train')   # Training masks
... val_mask_dir = os.path.join(dataset_dir, 'masks', 'val')       # Validation masks
... test_mask_dir = os.path.join(dataset_dir, 'masks', 'test')     # Test masks
... 
... # Create directories if they do not exist
... for dir_path in [train_img_dir, val_img_dir, test_img_dir, train_mask_dir, val_mask_dir, test_mask_dir]:
...     os.makedirs(dir_path, exist_ok=True)
... 
... # Debugging function to list files in directories
... def list_files_in_directory(directory):
...     print(f"Listing files in directory: {directory}")
...     for root, dirs, files in os.walk(directory):
...         for file in files:
...             print(os.path.join(root, file))
... 
... # List files in the original image and mask directories for debugging
... list_files_in_directory(image_dir)
... list_files_in_directory(mask_dir)
... 
... # Function to split and move files
... def split_and_move_files(img_dir, mask_dir, train_img_dir, val_img_dir, test_img_dir, train_mask_dir, val_mask_dir, test_mask_dir, train_size=0.7, val_size=0.2, test_size=0.1):
...     img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.h5')]  # Adjust the file extension as necessary
...     if len(img_files) == 0:
...         print(f"No images found in {img_dir}")
...         return
...     
...     # Split files into training, validation, and test sets
    train_files, temp_files = train_test_split(img_files, train_size=train_size, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_size/(val_size+test_size), random_state=42)
    
    # Move files to respective directories
    for file_set, dst_img_dir, dst_mask_dir in zip([train_files, val_files, test_files], [train_img_dir, val_img_dir, test_img_dir], [train_mask_dir, val_mask_dir, test_mask_dir]):
        for file in file_set:
            img_src = os.path.normpath(os.path.join(img_dir, file))
            img_dst = os.path.normpath(os.path.join(dst_img_dir, file))
            shutil.copy(img_src, img_dst)  # Copy image
            
            # Assuming mask file naming follows the pattern '<image_file_name>_mask.h5'
            mask_file = file.replace('.h5', '_mask.h5')  # Adjust according to your mask file naming convention
            mask_src = os.path.normpath(os.path.join(mask_dir, mask_file))
            mask_dst = os.path.normpath(os.path.join(dst_mask_dir, mask_file))
            
            if os.path.exists(mask_src):
                shutil.copy(mask_src, mask_dst)  # Copy corresponding mask
            else:
                print(f"Mask file not found: {mask_src}. Skipping.")

# Split and move files for the dataset
split_and_move_files(
    image_dir,
    mask_dir,
    train_img_dir, val_img_dir, test_img_dir,
    train_mask_dir, val_mask_dir, test_mask_dir
)

print("Dataset splitting and organization complete.")
