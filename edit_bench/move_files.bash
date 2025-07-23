#!/bin/bash

# Define the source and destination directories
src_dir="/ccn2/u/khaiaw/Code/counterfactual_benchmark/assets/iccv/annotations/frame1_translate"
dest_dir="/ccn2/u/rmvenkat/data/iccv_obj_motion_dataset_100img"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop over all .hdf5 files in the source directory
for file in "$src_dir"/*.hdf5; do
    # Extract the base name without extension (e.g., "0067" from "0067.hdf5")
    base=$(basename "$file" .hdf5)
    # Remove leading zeros for arithmetic comparison
    num=$(echo "$base" | sed 's/^0*//')

    # Check if the numeric value is 68 or above
    if [ "$num" -ge 68 ]; then
        # Create the new name by appending _translate before the extension
        newname="${base}_translate.hdf5"
        # Move and rename the file to the destination directory
        cp "$file" "$dest_dir/$newname"
        echo "Moved and renamed $file to $dest_dir/$newname"
    fi
done
