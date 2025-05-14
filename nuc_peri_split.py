# # when calling the script, use terminal below: 
# python nuc_peri_split.py -i /PATH-TO-MASKS-DIRECTORY

from skimage import io, morphology
import numpy as np
import glob
import os
import re
import warnings
import argparse

warnings.filterwarnings("ignore", 
                        category=UserWarning, 
                        message=".* is a low contrast image")

def nucleus_periphery_split(input_dir):

    size = 10 # length in pixels to trim all around 

    output_dir_interior = f"{input_dir}/interior"
    output_dir_periphery = f"{input_dir}/periphery"

    if(not os.path.exists(output_dir_interior)):
        os.mkdir(output_dir_interior)
        
    if(not os.path.exists(output_dir_periphery)):
        os.mkdir(output_dir_periphery)
        
    mask_files = glob.glob(f"{input_dir}/*_MASK.tif")
    for file_ in mask_files:
        file_name = os.path.split(file_)[1]
        print(f"Processing {file_name}")
        
        nucleus_mask = io.imread(file_)
        interior_mask = morphology.erosion(nucleus_mask, morphology.disk(size))
        periphery_mask = nucleus_mask - interior_mask
        
        # Check all is as expected - we should have same number of labels in each mask image
        if not (len(np.unique(nucleus_mask)) == 
                len(np.unique(interior_mask)) == 
                len(np.unique(periphery_mask))
            ):
            print(f"Error: Number of labels in the masks do not match for file {file_name}.")
                
        # Save the new masks to separate folders                                          
        io.imsave(f"{output_dir_interior}/{file_name}", interior_mask)
        io.imsave(f"{output_dir_periphery}/{file_name}", periphery_mask)
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    requiredGrp = ap.add_argument_group('required arguments')
    requiredGrp.add_argument("-i",'--input_dir', required=True, help="masks folder location")
    args = vars(ap.parse_args())
    input_dir = args['input_dir']

nucleus_periphery_split(input_dir)