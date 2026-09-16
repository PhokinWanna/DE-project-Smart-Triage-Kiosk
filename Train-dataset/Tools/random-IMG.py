import os
import random
import shutil

def pick_random_images(source_dir, dest_dir, num_files_to_pick):
    # Supported image extensions
    extensions = ('.jpg', '.jpeg', '.png')
    all_images = []

    # 1. Search for all matching images in the source folder
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(extensions):
                all_images.append(os.path.join(root, file))

    total_found = len(all_images)
    print(f"Found {total_found} images in '{source_dir}'.")

    if total_found == 0:
        print("No images found. Exiting.")
        return

    # If you ask for more images than exist, just grab all of them
    if num_files_to_pick > total_found:
        print(f"Requested {num_files_to_pick} images, but only found {total_found}. Copying all {total_found}.")
        num_files_to_pick = total_found

    # 2. Randomly select the images
    selected_images = random.sample(all_images, num_files_to_pick)

    # 3. Create the destination folder if it doesn't already exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 4. Copy the selected files to the new folder
    print(f"Copying {num_files_to_pick} images to '{dest_dir}'...")
    
    for image_path in selected_images:
        filename = os.path.basename(image_path)
        dest_path = os.path.join(dest_dir, filename)

        # Prevent overwriting if two files have the exact same name
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            # Add a random number to the end of the filename
            dest_path = os.path.join(dest_dir, f"{base}_{random.randint(1000,9999)}{ext}")

        # shutil.copy2 copies the file along with its metadata (like creation date)
        shutil.copy2(image_path, dest_path)

    print("Done! Check your destination folder.")

# ==========================================
# MODIFY THESE VARIABLES BEFORE RUNNING
# ==========================================
SOURCE_FOLDER = r"E:\SmartTriageAI\DataSet-Archive\Redness\Skin-Problem-Detection-Relabel-Clean3.v1i.multiclass\train" 
DESTINATION_FOLDER = r"E:\SmartTriageAI\Train-dataset\dataset\Flushing"
NUMBER_OF_IMAGES = 1250  # Change this to the number of images you want to pick
# ==========================================

if __name__ == "__main__":
    pick_random_images(SOURCE_FOLDER, DESTINATION_FOLDER, NUMBER_OF_IMAGES)