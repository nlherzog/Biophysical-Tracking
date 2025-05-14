# To run, type in command line: python combine_pre-post.py -i /path/to/input_folder -o /path/to/output_folder

import os
import re
import tifffile as tiff
import numpy as np
import argparse

def get_file_groups(folder):
    """Organizes TIFF files into groups based on their base name and last three-digit suffix."""
    file_dict = {}
    pattern = re.compile(r"(.+?)preTFH(\d{3})\.tif$")
    
    for file in os.listdir(folder):
        print(f"Checking file: {file}")  # Debug print to verify
        match = pattern.search(file)
        if match:
            base_name, suffix = match.groups()
            suffix = int(suffix)  # Convert to integer for sorting
            
            # Group files by their base name
            file_dict.setdefault(base_name, {}).setdefault(suffix, []).append(os.path.join(folder, file))
    
    # Sort the file groups to ensure proper merging
    sorted_file_dict = {}

    for base_name, suffix_dict in file_dict.items():
        sorted_suffixes = sorted(suffix_dict.keys())  # Sort the suffixes
        sorted_groups = []
        
        # Group the 'pre' and 'on' pairs by suffix
        for suffix in sorted_suffixes:
            pre_file = suffix_dict[suffix][0]  # Only one 'pre' file per suffix
            on_file = pre_file.replace("preTFH", "onTFH")  # Match the corresponding 'on' file
            
            if os.path.exists(on_file):  # Ensure both files exist
                sorted_groups.append((suffix, pre_file, on_file))
        
        sorted_file_dict[base_name] = sorted_groups
    
    print(f"Sorted file groups: {sorted_file_dict}")  # Debug print
    return sorted_file_dict

def concatenate_tiffs(file_groups, output_folder, channel_mapping):
    """Creates a hyperstack with onTFH as the main time series and preTFH inserted as t=0."""
    os.makedirs(output_folder, exist_ok=True)

    for base_name, suffix_groups in file_groups.items():
        for suffix, pre_file, on_file in suffix_groups:
            if not os.path.exists(on_file):
                print(f"Skipping {pre_file}, missing corresponding {on_file}")
                continue

            # Read the TIFF images
            pre_image = tiff.imread(pre_file)  # (Z, C, X, Y)
            on_image = tiff.imread(on_file)  # (T, Z, C, X, Y)

            print(f"Shape of {pre_file}: {pre_image.shape}")  # Expecting (Z, C, X, Y)
            print(f"Shape of {on_file}: {on_image.shape}")  # Expecting (T, Z, C, X, Y)

            # Ensure onTFH has a time dimension
            if len(on_image.shape) == 4:  
                on_image = np.expand_dims(on_image, axis=0)  # (1, Z, C, X, Y)

            T, Z, C_on, X, Y = on_image.shape  # Extract dimensions from onTFH

            # Create an empty array for preTFH with the same spatial dimensions
            pre_selected = np.zeros((1, Z, C_on, X, Y), dtype=on_image.dtype)  

            # Map the specified channels from preTFH to their correct positions
            for pre_ch, on_ch in channel_mapping.items():
                pre_selected[:, :, on_ch-1, :, :] = pre_image[:, pre_ch-1, :, :]  # Insert preTFH channel at correct index

            # Stack preTFH (T=0) and onTFH (T=1+)
            final_hyperstack = np.concatenate((pre_selected, on_image), axis=0)  # (T+1, Z, C, X, Y)

            # Save as a hyperstack
            output_filename = f"{base_name}allTFH{str(suffix).zfill(3)}.tif"
            output_file = os.path.join(output_folder, output_filename)
            tiff.imwrite(output_file, final_hyperstack, imagej=True)  # Ensure ImageJ compatibility

            print(f"Saved: {output_file}, shape: {final_hyperstack.shape}")



if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    requiredGrp = ap.add_argument_group('required arguments')
    requiredGrp.add_argument("-i", '--input_folder', required=True, help="parent folder location")
    requiredGrp.add_argument("-o", '--output_folder', required=True, help="parent folder location")
    args = vars(ap.parse_args())
    input_folder = args['input_folder']
    output_folder = args['output_folder']
    
    # Define the channel mapping (preTFH ch1 = onTFH ch3, preTFH ch4 = onTFH ch1)
    channel_mapping = {
        1: 1  # preTFH channel 4 maps to onTFH channel 2
        #4: 2,   # preTFH channel 6 maps to onTFH channel 4
        #6: 4   # preTFH channel 6 maps to onTFH channel 4
    }
    
    file_groups = get_file_groups(input_folder)
    concatenate_tiffs(file_groups, output_folder, channel_mapping)
