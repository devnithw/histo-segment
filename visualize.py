import openslide
import numpy as np
from PIL import Image

def save_wsi(img):
    slide = openslide.OpenSlide(img)
    # Avoid requesting full-resolution thumbnail (can exhaust memory).
    max_side = 2048
    w, h = slide.dimensions
    scale = max(w, h) / max_side if max(w, h) > max_side else 1
    thumb_size = (max(1, int(w / scale)), max(1, int(h / scale)))
    thumb = slide.get_thumbnail(thumb_size)
    thumb.save('thumbnail.png')

def save_mask(mask):
    print(f"Opening {mask}...")
    slide = openslide.OpenSlide(mask)
    print(f"Levels: {slide.level_count}, Dimensions: {slide.dimensions}")
    # Use lowest resolution level to avoid size issues
    level = slide.level_count - 1
    w, h = slide.level_dimensions[level]
    print(f"Reading level {level} with size {w}x{h}...")
    region = slide.read_region((0, 0), level, (w, h))
    
    # Convert to numpy array
    img_array = np.array(region)
    print(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
    print(f"Value range: min={img_array.min()}, max={img_array.max()}")
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array[:, :, :3], axis=2)
    
    # Min-max normalization to 0-255
    min_val = img_array.min()
    max_val = img_array.max()
    if max_val > min_val:
        normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(img_array, dtype=np.uint8)
    
    print(f"Normalized range: min={normalized.min()}, max={normalized.max()}")
    
    # Save
    img_pil = Image.fromarray(normalized, mode='L')
    img_pil.save('mask1_46.png')
    print(f"Saved to mask.png")


if __name__ == "__main__":
    img_id = "test_001"
    # img = f"/home/nadun/wd/datasets/camelyon16/test/masks/{img_id}_mask.tif"
    # save_wsi(img)
    mask = f"/home/nadun/wd/segmentation/inference_results/masks/test_001_pred_mask.tif"
    # m1 = '/home/nadun/wd/datasets/camelyon16/test/masks/test_046_mask.tif'
    # im46 = '/home/nadun/wd/datasets/camelyon16/test/images/test_046.tif'
    # save_wsi(im46)
    save_mask(mask)