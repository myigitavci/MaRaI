# Import required libraries

import os
import json
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
from tqdm import tqdm
import re
import csv
import glob

# ### Execute the Complete Pipeline
# 
# Set your input directory and the script will create a derivative folder for all outputs:

# %%
# Define the input directory containing NIfTI and JSON files
input_directory = "path/to/data"  # Change this to your input directory

# Create derivative directory structure
derivative_dir = os.path.join(os.path.dirname(input_directory), "derivative")
png_dir = os.path.join(derivative_dir, "png_with_planes")
csv_dir = os.path.join(derivative_dir, "csv_batches")
labeled_dir = os.path.join(derivative_dir, "labeled_data")
splits_dir = os.path.join(derivative_dir, "splits")

# Create all necessary directories
for directory in [png_dir, csv_dir, labeled_dir, splits_dir]:
    os.makedirs(directory, exist_ok=True)

# Define output file paths
bin_intervals_file = './MR-CLIP/bin_intervals_et_20_rt_20.json'  # Adapt this path as needed
final_train_file = os.path.join(splits_dir, "mr_train_with_planes_shuffled_slc_120_to_220.csv")
final_val_file = os.path.join(splits_dir, "mr_val_with_planes_shuffled_slc_120_to_220.csv")
final_test_file = os.path.join(splits_dir, "mr_test_with_planes_shuffled_slc_120_to_220.csv")

# ## Step 1: NIfTI to PNG Conversion
# 
# This step converts NIfTI files to PNG images while:
# - Preserving the folder structure
# - Determining the scanning plane (axial, coronal, sagittal)
# - Normalizing pixel values to 0-255
# - Saving only nonzero slices
# - Maintaining consistent naming convention

# %%
def determine_plane(nifti_path):
    """
    Determine the scanning plane from the NIfTI file's header information.
    Uses the pixel dimensions to identify the primary scanning plane.
    
    Args:
        nifti_path (str): Path to the NIfTI file.
    
    Returns:
        str: One of 'axial', 'coronal', or 'sagittal'.
    """
    try:
        nifti_img = nib.load(nifti_path)  
        index = np.argmax([0, nifti_img.header['pixdim'][1], 
                          nifti_img.header['pixdim'][2],
                          nifti_img.header['pixdim'][3]])
        if index == 1:
            return "sagittal"
        elif index == 2:
            return "coronal"
        elif index == 3:
            return "axial"
        else:
            print('Cannot decide the plane. Marking as axial')
            return "axial"

    except Exception as e:
        print(f"Error reading {nifti_path}: {e}")
        return "axial"

# %%
def process_nifti(nifti_path, output_dir, plane):
    """
    Process a NIfTI file and save its slices as PNG images.
    - Normalizes data to 0-255 range
    - Adjusts axis based on scanning plane
    - Saves only nonzero slices with minimum dimensions
    
    Args:
        nifti_path (str): Path to the NIfTI file.
        output_dir (str): Directory to save the PNG images.
        plane (str): Scanning plane ('axial', 'coronal', or 'sagittal').
    """
    try:
        # Load and normalize NIfTI data
        nifti_img = nib.load(nifti_path)
        nifti_data = nifti_img.get_fdata()
        normalized_data = (nifti_data - nifti_data.min()) / (nifti_data.max() - nifti_data.min()) * 255
        normalized_data = normalized_data.astype(np.uint8)

        # Adjust axis based on the plane
        if plane == "coronal":
            normalized_data = np.transpose(normalized_data, (0, 2, 1))
        elif plane == "sagittal":
            normalized_data = np.transpose(normalized_data, (1, 2, 0))

        # Process each slice
        for slice_idx in range(normalized_data.shape[2]):
            slice_data = normalized_data[:, :, slice_idx]
            if np.any(slice_data) and normalized_data.shape[0] > 40 and normalized_data.shape[1] > 40:
                base_name = os.path.basename(nifti_path).replace(".nii", "").replace(".gz", "")
                save_path = os.path.join(output_dir, f"{base_name}_{plane}_slice{slice_idx}.png")
                save_png(slice_data, save_path)

    except Exception as e:
        print(f"Error processing {nifti_path}: {e}")

# %%
def save_png(slice_data, save_path):
    """
    Save a single slice as a PNG image.
    
    Args:
        slice_data (np.ndarray): 2D array of the slice data.
        save_path (str): Path to save the PNG image.
    """
    try:
        img = Image.fromarray(slice_data)
        img.save(save_path)
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error saving PNG {save_path}: {e}")

# %%
def traverse_and_convert(root_dir, output_root):
    """
    Traverse the folder structure and convert NIfTI files to PNGs.
    - Skips files starting with 'ur_'
    - Skips angiogram files
    - Maintains folder structure in output
    
    Args:
        root_dir (str): Root directory containing the folder structure.
        output_root (str): Root directory for saving PNGs.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("ur") and (file.endswith(".nii") or file.endswith(".nii.gz")) and "angio" not in file:
                nifti_path = os.path.join(subdir, file)

                # Derive the output path based on input folder structure
                relative_path = os.path.relpath(subdir, root_dir)
                output_dir = os.path.join(output_root, "png_with_planes", relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Determine the plane and process the file
                plane = determine_plane(nifti_path)
                process_nifti(nifti_path, output_dir, plane)
print("Step 1: Converting NIfTI files to PNG...")
traverse_and_convert(input_directory, derivative_dir)                

# ## Step 2: Create CSV with Image-Metadata Pairs
# 
# This step creates CSV files containing:
# - Image paths
# - Simplified and structured text descriptions
# - Metadata extracted from JSON files
# - Organized in batches for efficient processing

# %%
def simplify_text(input_str):
    """
    Create a structured text description from metadata string.
    Organizes information into categories:
    - Plane information
    - Patient demographics
    - Scanner details
    - Protocol information
    - Imaging parameters
    
    Args:
        input_str (str): Raw metadata string from JSON.
    
    Returns:
        str: Structured text description.
    """
    # Define expected tags grouped by category
    categories = {
        "Plane": ["Plane"],
        "Scanner": ["Manufacturer", "Manufacturers Model Name", "Magnetic Field Strength"],
        "Protocol": ["Series Description", "Scanning Sequence", "Sequence Variant"],
        "Parameters": ["Echo Time", "Repetition Time", "Inversion Time", "Flip Angle"],
    }

    # Initialize dictionary with "NONE" for all expected tags
    tag_values = {tag: "NONE" for group in categories.values() for tag in group}

    # Extract the "plane" value separately
    plane_match = re.search(r"plane (\w+)", input_str, re.IGNORECASE)
    tag_values["Plane"] = plane_match.group(1) if plane_match else "NONE"

    # Use regex to extract tag-value pairs
    pattern = re.compile(r"(\b" + r"\b|\b".join(tag_values.keys()) + r"\b)\s+([^,]+)")
    matches = pattern.findall(input_str)

    # Store extracted values in dictionary
    for tag, value in matches:
        tag_values[tag] = value.strip()

    # Construct ordered output
    plane_text = f"A brain MRI, plane {tag_values['Plane']}"
    scanner_text = f"Scanner (Manufacturer, Model, Field Strength): ({', '.join(tag_values[tag] for tag in categories['Scanner'])})"
    protocol_text = f"Acquisition (Description, Sequence, Variant): ({', '.join(tag_values[tag] for tag in categories['Protocol'])})"
    parameters_text = f"Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): ({', '.join(tag_values[tag] for tag in categories['Parameters'])})"

    return f"{plane_text}, {scanner_text}, {protocol_text}, {parameters_text}"

# %%
def generate_text_from_json(json_path, plane):
    """
    Generate a descriptive text string from a JSON file.
    Extracts specific DICOM tags and formats them into a readable string.
    
    Args:
        json_path (str): Path to the JSON file.
        plane (str): The imaging plane extracted from the PNG filename.
    
    Returns:
        str: Generated text string.
    """
    keys_to_include = [
        "MagneticFieldStrength",
        "Manufacturer",
        "ManufacturersModelName",
        "SeriesDescription",
        "MRAcquisitionType",
        "ScanningSequence",
        "SequenceVariant",
        "SliceThickness",
        "EchoTime",
        "RepetitionTime",
        "InversionTime",
        "FlipAngle",
    ]

    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)

        # Build the text description
        description_parts = [f"a photo of brain MRI, plane {plane},"]
        for key in keys_to_include:
            if key in json_data:
                value = json_data[key]
                readable_key = re.sub(r"(?<!^)(?=[A-Z])", " ", key)
                if isinstance(value, (int, float, str)):
                    description_parts.append(f"{readable_key} {value}")
                elif isinstance(value, list):
                    description_parts.append(f"{readable_key} {', '.join(map(str, value))}")

        return ", ".join(description_parts)

    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return None

# %%
def find_png_and_json_in_batches(png_root, rawdata_root, batch_size, output_dir):
    """
    Process PNG files and their corresponding JSON files to create CSV batches.
    - Matches PNG files with JSON files
    - Filters slices based on plane-specific ranges
    - Generates and simplifies text descriptions
    - Creates batched CSV files
    
    Args:
        png_root (str): Root directory containing PNG files.
        rawdata_root (str): Root directory containing raw JSON files.
        batch_size (int): Number of rows per CSV file.
        output_dir (str): Directory to save the CSV files.
    """
    batch_counter = 0
    file_counter = 0
    current_batch = []

    os.makedirs(output_dir, exist_ok=True)

    for subdir, dirs, files in os.walk(png_root):
        dirs.sort() 
        files.sort()

        for file in files:
            if file.endswith(".png"):
                # Extract the slice number and plane from the file name
                match = re.search(r"_slice(\d+)\.png$", file)
                if match:
                    slice_number = int(match.group(1))

                    # Determine the plane and slice range
                    if "axial" in file.lower():
                        plane = "axial"
                        slice_range = range(121, 221)
                    elif "coronal" in file.lower():
                        plane = "coronal"
                        slice_range = range(121, 221)
                    elif "sagittal" in file.lower():
                        plane = "sagittal"
                        slice_range = range(40, 161)
                    else:
                        plane = "unknown"
                        slice_range = range(121, 221)

                    # Process only if the slice number is within the plane's range
                    if slice_number in slice_range:
                        png_path = os.path.join(subdir, file)

                        # Derive the JSON path
                        relative_path = os.path.relpath(subdir, png_root)
                        json_name = file.split("_slice")[0].rsplit("_", 1)[0] + ".json"
                        json_path = os.path.join(rawdata_root, relative_path, json_name)
                        
                        # Handle `ur_` prefix in JSON file names
                        if not os.path.exists(json_path) and json_name.startswith("ur_"):
                            json_path = os.path.join(rawdata_root, relative_path, json_name[3:])

                        # Process if JSON exists
                        if os.path.exists(json_path):
                            # Generate and simplify text
                            raw_text = generate_text_from_json(json_path, plane)
                            if raw_text:
                                simplified_text = simplify_text(raw_text)
                                current_batch.append({
                                    "filepath": png_path,
                                    "text": simplified_text
                                })
                                file_counter += 1

                        # Write batch to CSV if batch size is reached
                        if file_counter >= batch_size:
                            batch_file = os.path.join(output_dir, f"image_metadata_pairs_batch_{batch_counter}.csv")
                            save_csv(current_batch, batch_file)
                            batch_counter += 1
                            file_counter = 0
                            current_batch = []

    # Write any remaining files to a final batch
    if current_batch:
        batch_file = os.path.join(output_dir, f"image_metadata_pairs_batch_{batch_counter}.csv")
        save_csv(current_batch, batch_file)

# %%
def save_csv(data, output_csv):
    """
    Save the data to a CSV file.
    
    Args:
        data (list): List of dictionaries containing `filepath` and `text`.
        output_csv (str): Path to the output CSV file.
    """
    try:
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["filepath", "text"])
            writer.writeheader()
            writer.writerows(data)
        print(f"Batch saved to: {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
print("\nStep 2: Creating CSV files with metadata...")
find_png_and_json_in_batches(png_dir, input_directory, batch_size=20000, output_dir=csv_dir)
# ## Step 3: Label Data with Clusters
# 
# This step processes the CSV files to:
# - Extract and parse DICOM metadata
# - Create unique labels for parameter combinations
# - Bin numerical values (Echo Time, Repetition Time, etc.)
# - Track label distributions

# %%
def get_bin_label(value, bins):
    """
    Assign a bin label to a numerical value based on predefined intervals.
    
    Args:
        value: The numerical value to bin.
        bins: List of dictionaries containing bin ranges and labels.
    
    Returns:
        str: The bin label for the value.
    """
    if value is not None:
        value = float(value) 
    else:
        return None
    for bin_info in bins:
        bin_range = bin_info['range'].split(' - ')
        if len(bin_range) == 2:
            lower_bound = float(bin_range[0])
            upper_bound = float(bin_range[1])
            if lower_bound < value <= upper_bound:
                return bin_info['bin']
        else:
            if value >= float(bin_range[0].replace('>', '')):
                return bin_info['bin']
    return None

# %%
def parse_dicom_metadata(text):
    """
    Parse DICOM metadata from text string using regex patterns.
    
    Args:
        text (str): Text string containing metadata.
    
    Returns:
        dict: Dictionary of parsed metadata.
    """
    patterns = {
        'plane': r'(?:plane|Plane)\s+(\S+)',
        'Magnetic Field Strength': r'Magnetic Field Strength\s+([\d\.]+)',
        'Manufacturer': r'Manufacturer\s+(\S+)',
        'Manufacturers Model Name': r'Manufacturers Model Name\s+([^,]+)',
        'Series Description': r'Series Description\s+([^,]+)',
        'Acquisition Type': r'Acquisition Type\s+(\S+)',
        'Scanning Sequence': r'Scanning Sequence\s+(\S+)',
        'Sequence Variant': r'Sequence Variant\s+(\S+)',
        'Slice Thickness': r'Slice Thickness\s+([\d\.]+)',
        'Echo Time': r'Echo Time\s+([\d\.]+)',
        'Repetition Time': r'Repetition Time\s+([\d\.]+)',
        'Flip Angle': r'Flip Angle\s+([\d\.]+)',
        'Inversion Time': r'Inversion Time\s+([\d\.]+)',
    }
    
    metadata = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        metadata[key] = match.group(1).strip() if match else None
        
    return metadata

# %%
def label_data(df, bin_intervals, label_dict, label_counter):
    """
    Create unique labels for each combination of parameters.
    - Normalizes manufacturer model names
    - Creates base labels from categorical parameters
    - Adds binned numerical values
    - Assigns unique numbers to each label combination
    
    Args:
        df (pd.DataFrame): Input DataFrame with metadata.
        bin_intervals (dict): Dictionary of bin intervals for numerical parameters.
        label_dict (dict): Dictionary mapping label strings to unique numbers.
        label_counter (int): Counter for unique label numbers.
    
    Returns:
        tuple: (Labeled DataFrame, updated label dictionary, updated counter)
    """
    labels = []
    model_name_mapping = {
        "SIGNA_HDx": "Signa_HDxt",
        "DISCOVERY_MR450": "DISCOVERY_MR",
        "DISCOVERY_MR750": "DISCOVERY_MR"
    }

    for index, row in df.iterrows():
        # Normalize the Manufacturers Model Name
        model_name = row['Manufacturers Model Name']
        if model_name in model_name_mapping:
            model_name = model_name_mapping[model_name]

        # Create base label from categorical parameters
        label = f"{row['plane']}{row['Manufacturer']}_{model_name}_{row['Acquisition Type']}_{row['Scanning Sequence']}_{row['Sequence Variant']}_{row['Magnetic Field Strength']}_{row['Flip Angle']}"
        
        # Get bin labels for numerical parameters
        echo_time_bin = get_bin_label(row['Echo Time'], bin_intervals['Echo Time'])
        repetition_time_bin = get_bin_label(row['Repetition Time'], bin_intervals['Repetition Time'])
        inversion_time_bin = get_bin_label(row['Inversion Time'], bin_intervals['Inversion Time'])
        
        # Append bin labels to main label
        label += f"_{echo_time_bin}_{repetition_time_bin}_{inversion_time_bin}"
        
        # Assign unique number to label
        if label not in label_dict:
            label_dict[label] = label_counter
            label_counter += 1
        
        labels.append(label_dict[label])

    df['label'] = labels
    return df, label_dict, label_counter

# %%
def process_and_label_batches(input_dir, output_dir, bin_intervals_file):
    """
    Process all CSV batches and create labeled datasets.
    - Loads bin intervals from JSON
    - Processes each batch
    - Tracks label distributions
    - Saves labeled data and label samples
    
    Args:
        input_dir (str): Directory containing input CSV batches.
        output_dir (str): Directory to save labeled datasets.
        bin_intervals_file (str): Path to JSON file containing bin intervals.
    """
    # Load bin intervals
    with open(bin_intervals_file, 'r') as f:
        bin_intervals = json.load(f)

    # Initialize label tracking
    global_label_dict = {}
    global_label_counter = 0
    label_samples = pd.DataFrame(columns=['label', 'sample', 'count'])

    # Process each batch
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_csv = os.path.join(input_dir, filename)
            print(f"Processing {filename}")
            
            # Read and process batch
            df = pd.read_csv(input_csv)
            metadata_dicts = df['text'].apply(parse_dicom_metadata)
            metadata_df = pd.DataFrame(metadata_dicts.tolist())
            df = pd.concat([df, metadata_df], axis=1)
            
            # Filter out specific Flip Angle values
            df = df[~df['Flip Angle'].isin([30.0, 120.0, 15.0])]
            
            # Label the data
            labeled_df, global_label_dict, global_label_counter = label_data(
                df, bin_intervals, global_label_dict, global_label_counter
            )
            
            # Save labeled batch
            output_file = os.path.join(output_dir, f'labeled_{filename}')
            labeled_df.to_csv(output_file, index=False)
            print(f"Labeled data saved to: {output_file}")
            
            # Update label samples
            label_counts = labeled_df['label'].value_counts().reset_index()
            label_counts.columns = ['label', 'count']
            label_samples_chunk = labeled_df.drop_duplicates(subset=['label']).merge(label_counts, on='label')
            
            # Update global label samples
            for _, row in label_samples_chunk.iterrows():
                if row['label'] in label_samples['label'].values:
                    label_samples.loc[label_samples['label'] == row['label'], 'count'] += row['count']
                else:
                    label_samples = pd.concat([
                        label_samples,
                        pd.DataFrame({
                            'label': [row['label']],
                            'sample': [row['text']],
                            'count': [row['count']]
                        })
                    ], ignore_index=True)

    # Save label samples
    label_samples_output_file = os.path.join(output_dir, 'label_samples.csv')
    label_samples.to_csv(label_samples_output_file, index=False)
    print(f"Label samples saved to: {label_samples_output_file}")
print("\nStep 3: Labeling data with clusters...")
process_and_label_batches(csv_dir, labeled_dir, bin_intervals_file)
# ## Step 4: Merge, Shuffle, and Split Data
# 
# This step:
# - Merges all labeled CSV files
# - Shuffles data while keeping slices together
# - Splits into train/val/test sets
# - Applies slice filtering
# - Maintains data organization

# %%
def extract_slice_number(filepath):
    """
    Extract the slice number from the filename.
    
    Args:
        filepath (str): Path to the image file.
    
    Returns:
        int: Slice number if found, None otherwise.
    """
    match = re.search(r"_slice(\d+)\.png$", filepath)
    return int(match.group(1)) if match else None

# %%
def extract_image_id(filepath):
    """
    Extract the unique image identifier before `_slice`.
    
    Args:
        filepath (str): Path to the image file.
    
    Returns:
        str: Image identifier or full path if no match.
    """
    match = re.match(r"(.*)_slice\d+\.png$", filepath)
    return match.group(1) if match else filepath

# %%
def filter_slices(df, col, min_slice=100, max_slice=200):
    """
    Filter rows based on slice number and plane conditions.
    - Applies different ranges for sagittal plane
    - Ensures even-numbered slices
    - Maintains plane-specific filtering
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column name containing file paths.
        min_slice (int): Minimum slice number.
        max_slice (int): Maximum slice number.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    def is_slice_in_range(row):
        slice_number = extract_slice_number(row[col])
        if slice_number is not None:
            if 'sagittal' in row[col]:
                return 50 <= slice_number <= 150 and slice_number % 2 == 0
            return min_slice <= slice_number <= max_slice and slice_number % 2 == 0
        return False

    return df[df.apply(is_slice_in_range, axis=1)]

# %%
def clean_text_columns(df):
    """
    Clean text columns and filter slices.
    Applies standard slice filtering (120-220) for all planes.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    return filter_slices(df, "filepath", min_slice=120, max_slice=220)

# %%
def merge_and_shuffle_split_csv(input_folder, train_file, val_file, test_file, train_ratio=0.6, val_ratio=0.1):
    """
    Merge all CSV files, shuffle the data while keeping slices together,
    and split into training, validation, and test sets.
    
    Args:
        input_folder (str): Directory containing input CSV files.
        train_file (str): Path to save training data.
        val_file (str): Path to save validation data.
        test_file (str): Path to save test data.
        train_ratio (float): Ratio of data to use for training.
        val_ratio (float): Ratio of data to use for validation.
    """
    # Get all CSV files
    csv_files = [file for file in glob.glob(os.path.join(input_folder, "labeled_*.csv"))]

    # Read and process each CSV file
    df_list = [clean_text_columns(pd.read_csv(file)) for file in csv_files]
    merged_df_label = pd.concat(df_list, ignore_index=True)

    # Extract unique image identifiers for grouping
    merged_df_label["image_id"] = merged_df_label["filepath"].apply(extract_image_id)
    # Keep only essential columns
    essential_columns = ['image_id','filepath', 'text', 'label']
    merged_df_label = merged_df_label[essential_columns]
    # Shuffle groups to ensure randomness while keeping slices together
    grouped = merged_df_label.groupby("image_id")
    shuffled_groups = grouped.apply(lambda x: x).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split data
    unique_ids = shuffled_groups["image_id"].unique()
    total_images = len(unique_ids)
    
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_ids = unique_ids[:train_end]
    val_ids = unique_ids[train_end:val_end]
    test_ids = unique_ids[val_end:]

    train_df = shuffled_groups[shuffled_groups["image_id"].isin(train_ids)]
    val_df = shuffled_groups[shuffled_groups["image_id"].isin(val_ids)]
    test_df = shuffled_groups[shuffled_groups["image_id"].isin(test_ids)]

    # Drop the auxiliary column
    train_df.drop(columns=["image_id"], inplace=True)
    val_df.drop(columns=["image_id"], inplace=True)
    test_df.drop(columns=["image_id"], inplace=True)

    # Save the datasets
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Training data saved to: {train_file} ({train_df.shape[0]} rows)")
    print(f"Validation data saved to: {val_file} ({val_df.shape[0]} rows)")
    print(f"Testing data saved to: {test_file} ({test_df.shape[0]} rows)")
print("\nStep 4: Merging, shuffling, and splitting data...")
merge_and_shuffle_split_csv(labeled_dir, final_train_file, final_val_file, final_test_file, 0.6, 0.1)