import os
import random
from PIL import Image, ImageDraw

image_size = (128, 128)
circle_radius = 10
num_images = 200

output_dir = 'fake_data'
os.makedirs(output_dir, exist_ok=True)

for i in range(num_images):
    image = Image.new('RGB', image_size, (255, 255, 255))
    
    draw = ImageDraw.Draw(image)
    
    x = random.randint(0, image_size[0] - 2 * circle_radius)
    y = random.randint(0, image_size[1] - 2 * circle_radius)
    
    circle_bbox = [x, y, x + 2 * circle_radius, y + 2 * circle_radius]
    
    draw.ellipse(circle_bbox, fill=(255, 0, 0))
    
    image_filename = os.path.join(output_dir, f'{i:03}.png')
    image.save(image_filename)

print(f"{num_images} images saved in the '{output_dir}' directory.")
