from src.visualize import load_image, load_mask, show_image_with_mask

# Example image and mask paths
image_path = "data/example_image.png"
mask_path = "data/example_mask.png"

# Load and visualize
image = load_image(image_path)
mask = load_mask(mask_path)
show_image_with_mask(image, mask, method='matplotlib')  # or 'opencv'
