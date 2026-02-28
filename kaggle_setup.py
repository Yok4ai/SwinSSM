import os
import glob
import json


def prepare_brats_data(input_dir, output_dir):
    """Scan BraTS case directories and write dataset.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(input_dir, list):
        file_paths = []
        for d in input_dir:
            file_paths.extend(glob.glob(os.path.join(d, '*')))
    else:
        file_paths = glob.glob(os.path.join(input_dir, '*'))

    file_paths.sort()

    def find_file(case_dir, base_name, suffix):
        """Return the first existing file matching the given suffix, trying common BraTS layouts."""
        # T2w: BraTS 2023 uses -t2w, BraTS 2021 uses -t2
        suffixes = ['-t2w', '-t2'] if suffix == '-t2w' else [suffix]
        for suf in suffixes:
            for ext in ['.nii.gz', '.nii']:
                fname = base_name + suf + ext
                # Standard flat layout
                p = os.path.join(case_dir, fname)
                if os.path.exists(p):
                    return p
                # Kaggle nested layout: case/file.nii/file.nii
                p = os.path.join(case_dir, fname, fname)
                if os.path.exists(p):
                    return p
        return os.path.join(case_dir, base_name + suffix + '.nii')

    def is_valid(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    file_list = []
    skipped = []

    for case_dir in file_paths:
        name = os.path.basename(case_dir)
        images = [
            find_file(case_dir, name, '-t1c'),
            find_file(case_dir, name, '-t1n'),
            find_file(case_dir, name, '-t2f'),
            find_file(case_dir, name, '-t2w'),
        ]
        label = find_file(case_dir, name, '-seg')
        missing = [f for f in images + [label] if not is_valid(f)]

        if not missing:
            file_list.append({"image": images, "label": label})
            if len(file_list) <= 10:
                print(f"  Added: {name}")
        else:
            skipped.append(name)
            if len(skipped) <= 5:
                print(f"  Skipped: {name} ({len(missing)} missing files)")
            elif len(skipped) == 6:
                print("  (suppressing further warnings)")

    print(f"\n{len(file_list)} valid cases, {len(skipped)} skipped")

    output_path = os.path.join(output_dir, "dataset.json")
    with open(output_path, 'w') as f:
        json.dump({"training": file_list}, f, indent=4)
    print(f"Wrote dataset.json -> {output_path}")
    return output_dir


def setup_kaggle_notebook(dataset="brats2023"):
    """Prepare data from Kaggle input directories."""
    if dataset == "brats2021":
        input_dir = '/kaggle/input/brats21'
    elif dataset == "combined":
        input_dir = ['/kaggle/input/brats21', '/kaggle/input/brats2023-part-1', '/kaggle/input/brats2023-part-2zip']
    else:
        input_dir = ['/kaggle/input/brats2023-part-1', '/kaggle/input/brats2023-part-2zip']
    return prepare_brats_data(input_dir, '/kaggle/working')


def setup_local_data(custom_input_dir):
    """Prepare data from a user-supplied path. Skips preparation if dataset.json already exists."""
    input_dir = custom_input_dir
    output_dir = 'dataset'
    dataset_json = os.path.join(output_dir, "dataset.json")

    if os.path.exists(dataset_json):
        print(f"dataset.json already exists at {dataset_json}, skipping preparation.")
        return output_dir

    dirs = input_dir if isinstance(input_dir, list) else [input_dir]
    if not all(os.path.exists(d) for d in dirs):
        print(f"Data not found at: {input_dir}")
        return None

    return prepare_brats_data(input_dir, output_dir)


if __name__ == "__main__":
    if os.path.exists('/kaggle/input'):
        output_dir = setup_kaggle_notebook()
    else:
        print("Provide a data path via --data_dir when running locally.")
        output_dir = None

    if output_dir:
        print(f"Setup complete. Dataset at: {output_dir}")
    else:
        print("Setup failed.")
