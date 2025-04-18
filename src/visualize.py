import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load a grayscale image."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def load_mask(mask_path):
    """Load a binary mask."""
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay mask with a color and transparency on image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask_colored = np.zeros_like(image_rgb)
    mask_colored[mask > 0] = color
    return cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)

def show_image_with_mask(image, mask, method='matplotlib'):
    """Display overlay using matplotlib or OpenCV."""
    overlay = overlay_mask(image, mask)
    if method == 'matplotlib':
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Segmentation Overlay")
        plt.axis("off")
        plt.show()
    elif method == 'opencv':
        cv2.imshow("Segmentation Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise ValueError("method must be 'matplotlib' or 'opencv'")
