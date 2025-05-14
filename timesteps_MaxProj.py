# when calling the script, use terminal below: 
# python3 timesteps_MaxProj.py -i /PATH-TO-INPUT-DIRECTORY -o /PATH-TO-OUTPUT-DIRECTORY -f IMAGE-FRAME-RATE

from tifffile import imread, imwrite
import numpy as np
from skimage import io
import re
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

def change_timesteps(input_folder, output_folder, framerate):
    print("Script started")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Looking for .tif files in: {input_folder}")

    for file_ in os.listdir(input_folder):
        if file_.lower().endswith(('.tif', '.tiff')):
            print(f"Processing {file_}")
            full_path = os.path.join(input_folder, file_)
            img = io.imread(full_path)  # shape: (time, height, width)

            for exposure in [1, 2, 4, 10]:
                img_list = []
                for start in range(0, len(img), exposure):
                    img_list.append(np.max(img[start:start+exposure, :, :], axis=0))
                img2 = np.stack(img_list)

                fileroot = os.path.splitext(os.path.splitext(file_)[0])[0]
                out_path = os.path.join(output_folder, f"{fileroot}_{framerate * exposure}ms.tif")
                io.imsave(out_path, img2)
                print(f"Saved {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    requiredGrp = ap.add_argument_group('required arguments')
    requiredGrp.add_argument("-i",'--input_folder', required=True, help="input folder location")
    requiredGrp.add_argument("-o",'--output_folder', required=True, help="output folder location")
    requiredGrp.add_argument("-f",'--framerate', required=True, help="image framerate")
    args = vars(ap.parse_args())
    input_folder = args['input_folder']
    output_folder = args['output_folder'] 
    framerate = int(args['framerate'])

#framerate = 25 ms #this is where I would change the frame rate for each new experiment if framerate has changed  

change_timesteps(input_folder,output_folder,framerate)