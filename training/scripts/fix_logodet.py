import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# Setup
TRAINING_DIR = Path(__file__).resolve().parent.parent # C:/Projects/descry/brand-vision/training (relative to script)
DATASET_DIR  = TRAINING_DIR / "datasets"
LOGODET_ROOT = DATASET_DIR / "logodet3k" / "LogoDet-3K"

# Clean destination folders to prevent merging issues
for d in ['brand_logo_split', 'yolo_logo_detection']:
    if (DATASET_DIR / d).exists():
        shutil.rmtree(DATASET_DIR / d)

TARGET_BRANDS = [
    'Apple', 'Canon', 'Nikon', 'ASUS', 'BOSE', 'Casio', 'Rolex',
    'Converse', 'lacoste', 'timberland', 'tommy hilfiger',
    'polo ralph lauren', 'under armour', 'the north face', 'patagoniaa',
    'oakley-1', 'oakley-2', 'louis vuitton-1', 'louis vuitton-2',
    'versacee', 'Armani', 'levi', 'xiaomimei',
]

BRAND_LABEL_MAP = {
    'Apple': 'Apple', 'Canon': 'Canon', 'Nikon': 'Nikon', 'ASUS': 'ASUS', 
    'BOSE': 'Bose', 'Casio': 'Casio', 'Rolex': 'Rolex', 'Converse': 'Converse', 
    'lacoste': 'Lacoste', 'timberland': 'Timberland', 'tommy hilfiger': 'TommyHilfiger', 
    'polo ralph lauren': 'RalphLauren', 'under armour': 'UnderArmour', 
    'the north face': 'TheNorthFace', 'patagoniaa': 'Patagonia', 
    'oakley-1': 'Oakley', 'oakley-2': 'Oakley', 'louis vuitton-1': 'LouisVuitton', 
    'louis vuitton-2': 'LouisVuitton', 'versacee': 'Versace', 'Armani': 'Armani', 
    'levi': 'Levis', 'xiaomimei': 'Xiaomi',
}

# Find brand folders
brand_paths = {}
for cat_dir in LOGODET_ROOT.iterdir():
    if not cat_dir.is_dir(): continue
    for brand_dir in cat_dir.iterdir():
        if brand_dir.is_dir() and brand_dir.name in TARGET_BRANDS:
            brand_paths[brand_dir.name] = brand_dir

matched = list(brand_paths.keys())
print(f"Matched {len(matched)} brands")

BRAND_SPLIT_DIR = DATASET_DIR / 'brand_logo_split'
YOLO_DIR = DATASET_DIR / 'yolo_logo_detection'

for split in ('train', 'val'):
    (YOLO_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

total_crops = 0
yolo_train = 0
yolo_val = 0

for folder_name in matched:
    brand_dir = brand_paths[folder_name]
    label = BRAND_LABEL_MAP[folder_name]
    
    xmls = list(brand_dir.glob('*.xml'))
    random.shuffle(xmls)
    n = len(xmls)
    
    train_xmls = xmls[:int(n*0.7)]
    val_xmls   = xmls[int(n*0.7):int(n*0.9)]
    test_xmls  = xmls[int(n*0.9):]
    
    for split_name, files in [('train', train_xmls), ('val', val_xmls), ('test', test_xmls)]:
        dest_dir = BRAND_SPLIT_DIR / split_name / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        yolo_type = 'train' if split_name == 'train' else 'val'
        
        for xml in files:
            img_path = brand_dir / (xml.stem + '.jpg')
            if not img_path.exists(): img_path = brand_dir / (xml.stem + '.png')
            if not img_path.exists(): continue
            
            # Parse XML
            try:
                tree = ET.parse(xml)
                root = tree.getroot()
                size = root.find('size')
                w = float(size.find('width').text)
                h = float(size.find('height').text)
            except:
                continue
                
            yolo_labels = []
            valid_crop = False
            
            try:
                img = Image.open(img_path).convert('RGB')
                im_w, im_h = img.size
            except:
                continue
                
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # YOLO format
                cx = (xmin + xmax) / 2.0 / im_w
                cy = (ymin + ymax) / 2.0 / im_h
                bw = (xmax - xmin) / im_w
                bh = (ymax - ymin) / im_h
                yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
                # Crop logic
                px, py = int((xmax-xmin)*0.1), int((ymax-ymin)*0.1)
                crop = img.crop((max(0, xmin-px), max(0, ymin-py), min(im_w, xmax+px), min(im_h, ymax+py)))
                if crop.width > 10 and crop.height > 10:
                    crop.save(dest_dir / f"{xml.stem}_{int(xmin)}_{int(ymin)}.jpg", 'JPEG', quality=90)
                    total_crops += 1
                    valid_crop = True
                    
            if valid_crop and split_name in ('train', 'val'):
                # Copy to YOLO
                pfx = f"{label}_{xml.stem}"
                shutil.copy(img_path, YOLO_DIR / 'images' / yolo_type / f"{pfx}{img_path.suffix}")
                with open(YOLO_DIR / 'labels' / yolo_type / f"{pfx}.txt", 'w') as f:
                    f.write('\n'.join(yolo_labels))
                if yolo_type == 'train': yolo_train += 1
                else: yolo_val += 1

print(f"Done! Crops: {total_crops}")
print(f"YOLO Train: {yolo_train}, YOLO Val: {yolo_val}")

yaml_text = f"path: {str(YOLO_DIR.resolve()).replace(chr(92), '/')}\ntrain: images/train\nval:   images/val\nnc: 1\nnames: ['logo']\n"
(YOLO_DIR / 'logo_data.yaml').write_text(yaml_text)
