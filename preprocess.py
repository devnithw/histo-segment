import multiresolutionimageinterface as mir
import numpy as np
from collections import Counter
import glob
import tqdm

def extract_masks(image_path, annotation_path, output_path):
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(image_path)
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(annotation_path)
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    camelyon17_type_mask = False
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 255, '_1': 255, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)


def unique_values_in_mask(mask_path):
    reader = mir.MultiResolutionImageReader()
    img = reader.open(mask_path)
    width, height = img.getDimensions()

    tile_size = 4096  
    print(f"mask dimensions: {width}x{height}, tile size: {tile_size}x{tile_size}")
    num0s = 0
    num1s = 0
    num2s = 0
    num255s = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)

            patch = img.getUCharPatch(x, y, w, h, 0)  # level 0
            patch = np.array(patch)
            num0s += np.sum(patch == 0)
            num1s += np.sum(patch == 1)
            num2s += np.sum(patch == 2)
            num255s += np.sum(patch == 255)
    return num0s, num1s, num2s, num255s

# map o to o, 1 to 127, 2 to 255
def modify_mask(mask_path, save_path):
    reader = mir.MultiResolutionImageReader()
    img = reader.open(mask_path)
    width, height = img.getDimensions()
    
    writer = mir.MultiResolutionImageWriter()
    writer.openFile(save_path)
    writer.setTileSize(512)
    writer.setCompression(mir.Compression_LZW)
    writer.setDataType(mir.DataType_UChar)
    writer.setColorType(mir.ColorType_Monochrome)
    writer.writeImageInformation(width, height)
    
    tile_size = 4096
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            
            patch = img.getUCharPatch(x, y, w, h, 0)
            patch = np.array(patch)
            
            patch[patch == 1] = 127
            patch[patch == 2] = 255
            
            writer.writeBaseImagePartToLocation(patch.flatten(), x, y)
    
    writer.finishImage()

import h5py, os
from PIL import Image

def patchify_masks(file_path, split='train'):
    file_name = os.path.basename(file_path).split('.')[0]
    coord_path = f"/home/nadun/wd/datasets/camelyon16/{split}/trident/20x_512px_0px_overlap/features_conch_v1_dual/{file_name}.h5"
    mask_path = f"/home/nadun/wd/datasets/camelyon16/{split}/masks/{file_name}_mask.tif"
    os.makedirs(f"/home/nadun/wd/datasets/camelyon16/{split}/patched_masks/{file_name}", exist_ok=True)
    coordinates = h5py.File(coord_path, 'r')['coords'][:]
    
    reader = mir.MultiResolutionImageReader()
    img = reader.open(mask_path)
    width, height = img.getDimensions()

    tile_size = 512
    # print(f"mask dimensions: {width}x{height}, tile size: {tile_size}x{tile_size}")
    
    for idx, (x, y) in enumerate(coordinates):
        w = int(min(tile_size, width - x))
        h = int(min(tile_size, height - y))

        patch = img.getUCharPatch(int(x), int(y), w, h, 0)  # level 0
        patch = np.array(patch).reshape((h, w))
        
        output_path = f"/home/nadun/wd/datasets/camelyon16/{split}/patched_masks/{file_name}/{idx}_{x}_{y}.png"
        Image.fromarray(patch).save(output_path)

if __name__ == "__main__":
    split = 'train'
    file_names = glob.glob(f"/home/nadun/wd/datasets/camelyon16/{split}/images/*.tif")
    for file_name in tqdm.tqdm(file_names):
        try:
            patchify_masks(file_name, split=split)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    print("Done patchifying masks.")
    # mask_path = "/home/nadun/wd/datasets/camelyon16/train/masks/normal_003_mask.tif"
    # num0s, num1s, num2s, num255s = unique_values_in_mask(mask_path)
    # print(f"Unique value counts in mask: 0s={num0s}, 1s={num1s}, 2s={num2s}, 255s={num255s}")