#!/bin/bash
ZIPFILE="/path/to/BRATS2021-training.zip"
OUTDIR="/path/to/BRATS2021_nii_gz"

mkdir -p "$OUTDIR"

unzip -Z1 "$ZIPFILE" | grep '\.nii$' | while read nii_file; do
    echo "Processing $nii_file"
    # Get just the patient directory (first level)
    patient_dir=$(echo "$nii_file" | cut -d'/' -f1)
    base_name=$(basename "$nii_file" .nii)
    # Create corresponding directory inside OUTDIR
    mkdir -p "$OUTDIR/$patient_dir"
    # Extract and compress the nii file, save as nii.gz keeping the name
    unzip -p "$ZIPFILE" "$nii_file" | gzip > "$OUTDIR/$patient_dir/$base_name.nii.gz"

done
