import h5py
import numpy as np

def inspect_h5_file(filepath):
    with h5py.File(filepath, 'r') as f:
        print("=== Keys in the file and their info ===")
        for key in f.keys():
            dataset = f[key]
            shape = dataset.shape if hasattr(dataset, 'shape') else 'N/A'
            dtype = dataset.dtype if hasattr(dataset, 'dtype') else 'N/A'
            print(f"Key: {key} | Shape: {shape} | Dtype: {dtype}")

        if 'split' in f:
            split_data = f['split'][:]
            print("\n=== 'split' key sample ===")
            print("First 10 values:", split_data[:10])
            unique_splits = np.unique(split_data)
            print("Unique split values:", unique_splits)
        else:
            print("\nNo key named 'split' found.")

        if 'split_GLIPunseen' in f:
            custom_split = f['split_GLIPunseen'][:]
            print("\n=== 'split_GLIPunseen' key sample ===")
            print("First 10 values:", custom_split[:10])
            unique_custom = np.unique(custom_split)
            print("Unique split_GLIPunseen values:", unique_custom)
        else:
            print("\nNo key named 'split_GLIPunseen' found.")

        box_key = "boxes_1024"
        if box_key in f:
            boxes = f[box_key][:]
            print(f"\n=== Bounding Boxes from '{box_key}' ===")
            print("Shape:", boxes.shape)
            print("First 5 boxes (format: [x1, y1, x2, y2]):")
            print(boxes[:5])
        else:
            print(f"\nNo key named '{box_key}' found.")

        if 'labels' in f:
            labels = f['labels'][:]
            print("\n=== Labels ===")
            print("Shape:", labels.shape)
            print("First 10 label entries:")
            print(labels[:10])
        else:
            print("\nNo key named 'labels' found.")

        if 'relationships' in f:
            relationships = f['relationships'][:]
            print("\n=== Relationships ===")
            print("Shape:", relationships.shape)
            print("First 5 relationships (each row is expected to be [sub_idx, obj_idx, predicate_idx]):")
            print(relationships[:5])
        else:
            print("\nNo key named 'relationships' found.")

        if 'predicates' in f:
            predicates = f['predicates'][:]
            print("\n=== Predicates ===")
            print("Shape:", predicates.shape)
            print("First 5 predicate entries:")
            print(predicates[:5])
        else:
            print("\nNo key named 'predicates' found.")

        if 'img_to_first_box' in f:
            print("\n=== 'img_to_first_box' ===")
            data = f['img_to_first_box'][:]
            print("Shape:", data.shape)
            print("First 10 entries:", data[:10])
        if 'img_to_last_box' in f:
            print("\n=== 'img_to_last_box' ===")
            data = f['img_to_last_box'][:]
            print("Shape:", data.shape)
            print("First 10 entries:", data[:10])
        if 'img_to_first_rel' in f:
            print("\n=== 'img_to_first_rel' ===")
            data = f['img_to_first_rel'][:]
            print("Shape:", data.shape)
            print("First 10 entries:", data[:10])
        if 'img_to_last_rel' in f:
            print("\n=== 'img_to_last_rel' ===")
            data = f['img_to_last_rel'][:]
            print("Shape:", data.shape)
            print("First 10 entries:", data[:10])

        print("\nInspection complete.")

if __name__ == '__main__':
    h5_file_path = './data/visual_genome/stanford_filtered/VG-SGG.h5'
    print("Inspecting file:", h5_file_path)
    inspect_h5_file(h5_file_path)