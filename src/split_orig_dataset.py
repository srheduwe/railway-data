import shutil
import random
import pandas as pd
from pathlib import Path

def split_original_dataset(
    source_base: str, 
    output_dir: str, 
    max_id: int = 315,
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    seed: int = 42
):
    source_path = Path(source_base)
    output_path = Path(output_dir)
    images_dir = source_path / "1 Images"
    excel_path = source_path / "classes.xlsx"
    
    # 1. Reproduce the EXACT SAME split as the previous script
    all_ids = list(range(1, max_id + 1))
    
    random.seed(seed)
    random.shuffle(all_ids)
    
    num_ids = len(all_ids)
    train_end = int(train_ratio * num_ids)
    val_end = train_end + int(val_ratio * num_ids)
    
    train_ids = set(all_ids[:train_end])
    val_ids = set(all_ids[train_end:val_end])
    test_ids = set(all_ids[val_end:])
    
    for split_name in ['train', 'val', 'test']:
        (output_path / split_name).mkdir(parents=True, exist_ok=True)

    print(f"Reading labels from {excel_path.name}...")
    
    # 2. Read the Excel file. 
    classes_df = pd.read_excel(excel_path, engine="openpyxl", header=None, names=["Class"])
    
    copied_count = {'train': 0, 'val': 0, 'test': 0}
    missing_count = 0

    print("Copying files to new directory structure...")

    # 3. Iterate through the rows in the DataFrame
    for row in classes_df.itertuples():
        img_id = row.Index + 1
        label = str(row.Class).strip()
        
        if img_id in train_ids:
            split = 'train'
        elif img_id in val_ids:
            split = 'val'
        elif img_id in test_ids:
            split = 'test'
        else:
            continue
            
        src_img = images_dir / f"{img_id}.jpg"
        
        if not src_img.is_file():
            print(f"Warning: Could not find image {src_img.name}. Skipping.")
            missing_count += 1
            continue
            
        dest_dir = output_path / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_img, dest_dir / src_img.name)
        copied_count[split] += 1

    print(f"Files copied -> Train: {copied_count['train']}, Val: {copied_count['val']}, Test: {copied_count['test']}")
    if missing_count > 0:
        print(f"Skipped {missing_count} missing files.")

if __name__ == "__main__":
    SOURCE_BASE = ""
    OUTPUT_DIRECTORY = ""
    
    split_original_dataset(
        source_base=SOURCE_BASE, 
        output_dir=OUTPUT_DIRECTORY,
        max_id=315
    )