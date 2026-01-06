import os
import sys
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm

# Add CONCH to the Python path
sys.path.insert(0, '/home/nadun/hdd/CONCH')

from conch.open_clip_custom import create_model_from_pretrained

def extract_features_for_slide(slide_id, images_dir, output_dir, model, preprocess, device='cuda'):
    # Define paths
    slide_folder = Path(images_dir) / slide_id
    output_folder = Path(output_dir) / slide_id
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all jpg files in the slide folder
    image_files = sorted(list(slide_folder.glob('*.jpg')))
    
    if not image_files:
        print(f"Warning: No .jpg files found in {slide_folder}")
        return
    
    print(f"\nProcessing slide: {slide_id}")
    print(f"Found {len(image_files)} patches")
    print(f"Output directory: {output_folder}")
    
    # Process each image
    for img_path in tqdm(image_files, desc=f"Extracting features for {slide_id}"):
        # Define output path (same name but .pt extension)
        output_path = output_folder / f"{img_path.stem}.pt"
        
        # Skip if already processed
        if output_path.exists():
            continue
        
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # Extract features
            with torch.inference_mode():
                features = model.encode_image(image_tensor)
            
            # Save features to disk (move to CPU first)
            torch.save(features.cpu(), output_path)
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue
    
    print(f"Feature extraction complete for {slide_id}")
    print(f"Saved {len(list(output_folder.glob('*.pt')))} feature files")


def main():
    # Configuration
    images_base_dir = '/home/nadun/wd/datasets/SICAP-test/images'
    output_base_dir = '/home/nadun/wd/datasets/SICAP-test/features'
    
    # CONCH model configuration
    model_cfg = 'conch_ViT-B-16'
    checkpoint_path = '/home/nadun/hdd/CONCH/checkpoints/conch/pytorch_model.bin'
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get all slide IDs from the images directory
    images_path = Path(images_base_dir)
    slide_ids = sorted([d.name for d in images_path.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(slide_ids)} slides to process")
    print("="*60)
    
    # Load CONCH model
    print("\nLoading CONCH model...")
    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Extract features for all slides
    print("\n" + "="*60)
    print("Starting feature extraction for all slides...")
    print("="*60)
    
    for idx, slide_id in enumerate(slide_ids, 1):
        print(f"\n[{idx}/{len(slide_ids)}] Processing slide: {slide_id}")
        
        extract_features_for_slide(
            slide_id=slide_id,
            images_dir=images_base_dir,
            output_dir=output_base_dir,
            model=model,
            preprocess=preprocess,
            device=device
        )
    
    print("\n" + "="*60)
    print("Feature extraction complete for all slides!")
    print(f"Processed {len(slide_ids)} slides")
    print(f"Features saved to: {output_base_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
