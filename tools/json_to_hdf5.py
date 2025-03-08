import json
import numpy as np
import h5py

def create_custom_h5(custom_json_path, output_h5_path):

    with open(custom_json_path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        custom_annotations = [data]
    elif isinstance(data, list):
        custom_annotations = data
    else:
        raise ValueError("Custom JSON file must be a dict or a list of dicts.")

    num_images = len(custom_annotations)
    print(f"Number of custom images: {num_images}")


    object_mapping = {"woman": 1, "man": 2}     
    predicate_mapping = {"on": 3, "with": 4}      


    all_boxes = []
    all_labels = []
    all_active = []  
    
    rel_pairs = []
    rel_preds = []

    img_to_first_box = []
    img_to_last_box = []
    img_to_first_rel = []
    img_to_last_rel = []

    for ann in custom_annotations:
        
        objects = ann.get("objects", [])
        num_objs = len(objects)

        img_to_first_box.append(len(all_boxes))
        img_to_last_box.append(len(all_boxes) + num_objs - 1)

        for obj in objects:
            bbox = obj.get("bbox", [0, 0, 1, 1])  #  [x1, y1, x2, y2]
            label_str = obj.get("label", "__background__")
            label = object_mapping.get(label_str, 0)
            all_boxes.append(bbox)
            all_labels.append([label])
            all_active.append([True])
        

        rels = ann.get("relationships", [])
        if rels:
            img_to_first_rel.append(len(rel_pairs))
            img_to_last_rel.append(len(rel_pairs) + len(rels) - 1)
            for r in rels:
                # {"subject": index, "object": index, "predicate": str}
                sub = r.get("subject", 0)
                obj = r.get("object", 0)
                pred_str = r.get("predicate", "[UNK]")
                pred = predicate_mapping.get(pred_str, 0)
                rel_pairs.append([sub, obj])
                rel_preds.append([pred])
        else:
            img_to_first_rel.append(-1)
            img_to_last_rel.append(-1)
    

    all_boxes = np.array(all_boxes, dtype=np.int32)                # shape (num_objs, 4)
    all_labels = np.array(all_labels, dtype=np.int64)              # shape (num_objs, 1)
    all_active = np.array(all_active, dtype=bool)                  # shape (num_objs, 1)
    rel_pairs = np.array(rel_pairs, dtype=np.int32)                # shape (num_rel, 2)
    rel_preds = np.array(rel_preds, dtype=np.int64)                # shape (num_rel, 1)

    img_to_first_box = np.array(img_to_first_box, dtype=np.int32)  # shape (num_images,)
    img_to_last_box = np.array(img_to_last_box, dtype=np.int32)    # shape (num_images,)
    img_to_first_rel = np.array(img_to_first_rel, dtype=np.int32)  # shape (num_images,)
    img_to_last_rel = np.array(img_to_last_rel, dtype=np.int32)    # shape (num_images,)


    split = np.full((num_images,), 2, dtype=np.int32)
    split_GLIPunseen = np.full((num_images,), 2, dtype=np.int32)
    split_custom = np.full((num_images,), 2, dtype=np.int32)


    with h5py.File(output_h5_path, "w") as f:
        f.create_dataset("active_object_mask", data=all_active, maxshape=all_active.shape, chunks=True)
        f.create_dataset("boxes_1024", data=all_boxes, maxshape=all_boxes.shape, chunks=True)
        f.create_dataset("boxes_512", data=all_boxes, maxshape=all_boxes.shape, chunks=True)
        f.create_dataset("img_to_first_box", data=img_to_first_box, maxshape=img_to_first_box.shape, chunks=True)
        f.create_dataset("img_to_last_box", data=img_to_last_box, maxshape=img_to_last_box.shape, chunks=True)
        f.create_dataset("img_to_first_rel", data=img_to_first_rel, maxshape=img_to_first_rel.shape, chunks=True)
        f.create_dataset("img_to_last_rel", data=img_to_last_rel, maxshape=img_to_last_rel.shape, chunks=True)
        f.create_dataset("labels", data=all_labels, maxshape=all_labels.shape, chunks=True)
        f.create_dataset("predicates", data=rel_preds, maxshape=rel_preds.shape, chunks=True)
        f.create_dataset("relationships", data=rel_pairs, maxshape=rel_pairs.shape, chunks=True)
        f.create_dataset("split", data=split, maxshape=split.shape, chunks=True)
        f.create_dataset("split_GLIPunseen", data=split_GLIPunseen, maxshape=split_GLIPunseen.shape, chunks=True)
        f.create_dataset("split_custom", data=split_custom, maxshape=split_custom.shape, chunks=True)
    
    print("Custom HDF5 file created at:", output_h5_path)

if __name__ == "__main__":
    custom_json = "./custom_data/custom_data.json"   
    output_h5 = "./data/visual_genome/stanford_filtered/custom_annotations.h5"
    create_custom_h5(custom_json, output_h5)