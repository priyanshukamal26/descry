# prepare_datasets.py
# Compiled from 01_data_preparation.ipynb for full-scale processing.

import os
import csv
import json
import shutil
import random
from collections import defaultdict
from pathlib import Path

def main():
    random.seed(42)

    # Use absolute paths to be safe
    TRAINING_DIR = Path(r"c:\Projects\descry\descry\training")
    DATASET_DIR  = TRAINING_DIR / 'datasets'
    LOGODET_ROOT = DATASET_DIR / 'logodet3k' / 'LogoDet-3K'
    LOGO2K_ROOT  = DATASET_DIR / 'logo2k' / 'datasetcopy' / 'trainandtest'
    FASHION_ROOT = DATASET_DIR / 'fashion_products'

    print(f'LogoDet root: {LOGODET_ROOT}')
    print(f'Logo2K+ root: {LOGO2K_ROOT}')
    print(f'Fashion root: {FASHION_ROOT}')

    # --- Cell 2: Build Brand Classifier Dataset ---
    print("\nProcessing Logo-2K+ for Brand Classifier...")
    BRAND_SPLIT_DIR = DATASET_DIR / 'brand_logo_split'
    if BRAND_SPLIT_DIR.exists(): 
        print(f"Removing existing {BRAND_SPLIT_DIR}...")
        shutil.rmtree(BRAND_SPLIT_DIR)

    total_logo2k_copied = 0
    classes_found = 0

    if LOGO2K_ROOT.exists():
        for split in ['train', 'test']:
            split_dir = LOGO2K_ROOT / split
            if not split_dir.exists(): continue
            
            dest_split = 'val' if split == 'test' else 'train'
            
            for root_cat in split_dir.iterdir():
                if not root_cat.is_dir(): continue
                for brand_dir in root_cat.iterdir():
                    if not brand_dir.is_dir(): continue
                    brand_name = brand_dir.name.replace(' ', '')
                    dest_dir = BRAND_SPLIT_DIR / dest_split / brand_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    if split == 'train': classes_found += 1
                    
                    for img_path in brand_dir.glob('*.*'):
                        if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                            try:
                                shutil.copy(img_path, dest_dir / img_path.name)
                                total_logo2k_copied += 1
                            except Exception as e:
                                print(f"Error copying {img_path}: {e}")

    print(f'Logo-2K+ Split: Copied {total_logo2k_copied} images across {classes_found} classes.')

    # --- Cell 3: Build YOLO Dataset ---
    print("\nProcessing LogoDet-3K for YOLO Logo Detection...")
    YOLO_DIR = DATASET_DIR / 'yolo_logo_detection'
    if YOLO_DIR.exists(): 
        print(f"Removing existing {YOLO_DIR}...")
        shutil.rmtree(YOLO_DIR)

    for split in ('train', 'val'):
        (YOLO_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    yolo_counts = {'train': 0, 'val': 0}
    logodet_classes = 0

    if LOGODET_ROOT.exists():
        for cat_dir in LOGODET_ROOT.iterdir():
            if not cat_dir.is_dir(): continue
            for brand_dir in cat_dir.iterdir():
                if not brand_dir.is_dir(): continue
                
                logodet_classes += 1
                brand_name = brand_dir.name.replace(' ', '')
                txts = list(brand_dir.glob('*.txt'))
                random.shuffle(txts)
                
                splits = {
                    'train': txts[:int(len(txts)*0.8)], 
                    'val': txts[int(len(txts)*0.8):]
                }
                
                for split_name, files in splits.items():
                    for txt in files:
                        img = brand_dir / (txt.stem + '.jpg')
                        if not img.exists(): img = brand_dir / (txt.stem + '.png')
                        if not img.exists(): continue
                        
                        try:
                            shutil.copy(img, YOLO_DIR / 'images' / split_name / f'{brand_name}_{txt.stem}{img.suffix}')
                            with open(txt) as fin, open(YOLO_DIR / 'labels' / split_name / f'{brand_name}_{txt.stem}.txt', 'w') as fout:
                                for line in fin:
                                    parts = line.strip().split()
                                    if parts:
                                        fout.write('0 ' + ' '.join(parts[1:]) + '\n')
                            yolo_counts[split_name] += 1
                        except Exception as e:
                            print(f"Error processing {img}: {e}")

    print(f"YOLO Dataset: Copied {yolo_counts['train']} train, {yolo_counts['val']} val across {logodet_classes} classes.")

    yaml_str = f"path: {str(YOLO_DIR.resolve()).replace(chr(92), '/')}\ntrain: images/train\nval:   images/val\nnc: 1\nnames: ['logo']\n"
    with open(YOLO_DIR / 'logo_data.yaml', 'w') as f:
        f.write(yaml_str)

    # --- Cell 4: Build Product Category Split ---
    print("\nProcessing Fashion Product Dataset for Category Classifier...")
    CAT_SPLIT_DIR = DATASET_DIR / 'product_category_split'
    if CAT_SPLIT_DIR.exists(): 
        print(f"Removing existing {CAT_SPLIT_DIR}...")
        shutil.rmtree(CAT_SPLIT_DIR)

    images_by_cat = defaultdict(list)
    csv_path = FASHION_ROOT / 'styles.csv'
    img_dir  = FASHION_ROOT / 'images'

    if csv_path.exists():
        with open(csv_path, encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                article_type = row.get('articleType', '').strip()
                img_id = row.get('id')
                if article_type and img_id:
                    clean_cat = article_type.replace('/', '_').replace('\\', '_')
                    images_by_cat[clean_cat].append(f'{img_id}.jpg')

    print(f"Found {len(images_by_cat)} unique fashion categories.")

    MIN_IMAGES_REQUIRED = 100
    valid_cats = {c: imgs for c, imgs in images_by_cat.items() if len(imgs) >= MIN_IMAGES_REQUIRED}
    print(f"Keeping {len(valid_cats)} sub-categories that have >{MIN_IMAGES_REQUIRED} images.")

    for ds in ('train', 'val', 'test'):
        for cat in valid_cats.keys():
            (CAT_SPLIT_DIR / ds / cat).mkdir(parents=True, exist_ok=True)

    total_fashion_copied = 0
    for cat, img_list in valid_cats.items():
        random.shuffle(img_list)
        n = min(len(img_list), 3000)
        img_list = img_list[:n]
        
        splits = {
            'train': img_list[:int(n*0.70)],
            'val':   img_list[int(n*0.70):int(n*0.90)],
            'test':  img_list[int(n*0.90):],
        }

        for split_name, files in splits.items():
            for f in files:
                src = img_dir / f
                if src.exists():
                    try:
                        shutil.copy(src, CAT_SPLIT_DIR / split_name / cat / f)
                        total_fashion_copied += 1
                    except Exception as e:
                        print(f"Error copying {src}: {e}")

    print(f'Fashion Category split done - {total_fashion_copied} total images copied.')
    print('\n✅ Phase 1 complete! Using ALL available classes.')

if __name__ == "__main__":
    main()
