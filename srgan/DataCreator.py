import os
import cv2
import numpy as np


def center_crop(image, crop_size=(1200, 900)):
    # Get the dimensions of the input image
    height, width, _ = image.shape

    # Calculate the top-left corner coordinates for center cropping
    top = max(0, (height - crop_size[1]) // 2)
    left = max(0, (width - crop_size[0]) // 2)

    # Perform the center crop
    cropped_image = image[top:top + crop_size[1], left:left + crop_size[0], :]

    return cropped_image


def mosaic(image, block_size=2):

    # Get the dimensions of the small image
    height, width, _ = image.shape

    # Calculate the number of blocks in both dimensions
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size

    # Resize the small image to a smaller size using pixelation
    small_mosaic_image = cv2.resize(image, (num_blocks_x, num_blocks_y), interpolation=cv2.INTER_NEAREST)

    # Resize the small mosaic image back to the original size
    mosaic_image = cv2.resize(small_mosaic_image, (width * block_size, height * block_size), interpolation=cv2.INTER_NEAREST)

    # Resize the image to one-fourth of its original size
    small_image = cv2.resize(mosaic_image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_LINEAR)

    return small_image

def add_random_noise(image, noise_intensity=0.1):
    # Generate random noise
    noise = np.random.normal(scale=noise_intensity, size=image.shape[:2]).astype(np.uint8)
    noise = np.stack([noise] * image.shape[2], axis=-1)

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)

    return noisy_image

def add_cloudy_effect(image, cloudiness=0.3):
    # Create a white overlay with transparency
    overlay = np.ones_like(image) * 255
    overlay = (cloudiness * overlay).astype(np.uint8)

    # Blend the image with the white overlay
    cloudy_image = cv2.addWeighted(image, 1 - cloudiness, overlay, cloudiness, 0)

    return cloudy_image


def createLrImage(image, crop_size=(1200,900), mosaic_intensity=2, noise_intensity=0.1, cloudiness=0.3):
    cropped = center_crop(image, crop_size)
    mosaic_image = mosaic(cropped, mosaic_intensity)
    mosaic_img_with_noise = add_random_noise(mosaic_image, noise_intensity)
    clouded_image = add_cloudy_effect(mosaic_img_with_noise, cloudiness)
    return clouded_image




if __name__ == "__main__":
    # Path to the folder containing high-resolution images
    input_folder = './Dataset/DIV2K_train_hr'

    # Output folder for storing downsampled, pixelated, and cloudy images
    output_folder = './Dataset/DIV2K_lr_5'
    os.makedirs(output_folder, exist_ok=True)

    # Desired parameters
    block_size = 5
    noise_intensity = 0.4
    cloudiness = 0.4

    # Loop through each image in the input folder
    for filename in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(input_image_path)
        lr_image = createLrImage(original_image, mosaic_intensity=5, noise_intensity=0.35, cloudiness=0.4)

        output_filename = f"{filename.split('_')[0]}_lr.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, lr_image)


# In[ ]:

