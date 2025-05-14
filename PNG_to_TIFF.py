import os
from PIL import Image

input_dir = '/Volumes/holtl02lab/holtl02labspace/Holt_Lab_Members/Nora_Holt/Live_Imaging_Other/250413_HeLa-2112-2113_2712-2713_6h-TFH/First_TFH_Masks'
output_dir = '/Volumes/holtl02lab/holtl02labspace/Holt_Lab_Members/Nora_Holt/Live_Imaging_Other/250413_HeLa-2112-2113_2712-2713_6h-TFH/First_TFH_Masks'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

for png_file in png_files:
    input_path = os.path.join(input_dir, png_file)
    output_path = os.path.join(output_dir, os.path.splitext(png_file)[0] + '.tif')

    png_image = Image.open(input_path)
    png_image.save(output_path, 'TIFF')
    png_image.close()
    print("conversion complete")