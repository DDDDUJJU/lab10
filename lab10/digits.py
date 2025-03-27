from PIL import Image, ImageDraw, ImageFont
import random
import os
import numpy as np

# Parameters
width, height = 8, 8
num_images = 10  # Number of images to generate

# Create a list to store all generated images
images = []

# Function: Generate a distorted digit image
def generate_distorted_image(digit):
    # Create an 8x8 pixel white background image
    img = Image.new('1', (width, height), 1)  # '1' represents binary (black and white) image mode
    
    # Get the drawing object
    draw = ImageDraw.Draw(img)
    
    # Use the default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 8)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw the digit on the image
    draw.text((0, 0), str(digit), font=font, fill=0)

    # Randomly translate the image (right, up, or down)
    dx = random.randint(-2, 0)  # Horizontal translation, up to 2 pixels to the left
    dy = random.randint(0, 2)  # Vertical translation, up to 2 pixels down

    # Perform the translation (affine transformation)
    img = img.transform((width, height), Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=1)  # Fill empty space with white

    return img

# Create a folder to store images
output_folder = './digits'
os.makedirs(output_folder, exist_ok=True)

# Generate multiple distorted digit images and save them as JPG files
for i in range(num_images):
    digit = random.choice([1, 2])  # Randomly choose 1 or 2
    img = generate_distorted_image(digit)
    
    # Save the image with the name as 'digit_index.jpg' (e.g., '1_1.jpg')
    img_path = os.path.join(output_folder, f'{digit}_{i+1}.jpg')
    img.save(img_path)

# Output the folder path where the generated images are stored
print(f"All generated images are saved to: {output_folder}")
