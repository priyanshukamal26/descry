# Descry: Full-Scale Training Playbook

Now that the system is configured to ingest **ALL 2,300+ brand classes** and **ALL fashion product classes** (extracting the maximum potential of your datasets), you need a structured approach to deploy this heavy compute pipeline. 

Because we are processing over 300,000 images combined, we will bypass local training limitations by preparing the data locally and offloading the actual training to Google Colab.

Follow these steps exactly to guarantee success.

---

## Phase 1: Local Data Preparation
*Goal: Process raw images into perfectly sorted class folders.*

1. **Prerequisites:** 
   Ensure `logodet3k/`, `logo2k/`, and `fashion_products/` are unzipped inside `descry/training/datasets/`.
2. **Launch Jupyter:** 
   Open `descry/training/01_data_preparation.ipynb` in your local Jupyter environment.
3. **Execute:** 
   Run all cells. 
   - *Note:* This will take some time. It is creating the optimal `train` and `val` splits for thousands of categories and copying over 300,000 images into `brand_logo_split` and `product_category_split`.
4. **Zip Outputs:** 
   Once the script finishes, navigating to `training/datasets/` and compress the output folders into zip files:
   - `brand_logo_split.zip`
   - `product_category_split.zip`
   - `yolo_logo_detection.zip`

---

## Phase 2: Cloud Storage Setup
*Goal: Move the processed splits to the cloud for fast GPU access.*

1. **Google Drive:** 
   Go to your Google Drive and create a directory named `descry_training`.
2. **Upload:** 
   Upload the three `.zip` files you created in Phase 1 into the `descry_training` folder.
   - *Pro Tip:* Keep your machine awake during the upload. Using the Google Drive Desktop app is faster than the browser.

---

## Phase 3: Colab Training Engine
*Goal: Train the models using free T4/L4 GPUs.*

### A. The Brand & Category Classifiers
1. **Open Colab:** Go to [colab.research.google.com](https://colab.research.google.com/) and create a new notebook.
2. **Mount Drive:** 
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Unzip Data:** Extract your zip directly in the Colab high-speed environment.
   ```python
   !unzip -q /content/drive/MyDrive/descry_training/brand_logo_split.zip -d /content/
   ```
4. **Copy Notebook Code:** Copy the Python code from `descry/training/04_train_efficientnet_brand.ipynb` into Colab cells.
5. **Update Paths & Train:** 
   Change `DATA_DIR = Path('/content/brand_logo_split')`. Set your runtime to **GPU (T4)** and run all cells.
6. **Save Weights:** After training, Colab will spit out `brand_classifier.onnx` and `brand_classifier_labels.json`. Save these back to your Google Drive!
   - Repeat this exact process for the Category Classifier using `03_train_efficientnet_category.ipynb` and the category zip.

### B. The YOLOv8 Logo Detector
1. **Unzip Data:** 
   ```python
   !unzip -q /content/drive/MyDrive/descry_training/yolo_logo_detection.zip -d /content/
   ```
2. **Copy Notebook Code:** Use the code from `02_train_yolo_logo.ipynb`.
3. **Train:** Run the YOLOv8 training. Since it's only detecting ONE class ("logo") across all images, it focuses purely on finding the logo boundaries.
4. **Export:** Grab the final `best.onnx` and save it to your Drive as `logo_detector.onnx`.

---

## Phase 4: System Integration
*Goal: Plug the massive brains into your sleek frontend UI.*

1. **Download Weights:** Fetch the `.onnx` and `.json` files from your Drive to your local PC.
2. **Deploy to Backend:** Place them exactly inside `descry/backend/weights/`:
   - `brand_classifier.onnx`
   - `brand_classifier_labels.json`
   - `category_classifier.onnx`
   - `category_classifier_labels.json`
   - `logo_detector.onnx`
3. **Reboot Engine:** 
   Restart your `start_servers.bat`. 
   The FastAPI backend reads the JSON label lists dynamically. This means your frontend will now natively recognize over 2,300 brands without any UI code changes!

---

**Success Checklist:**
- [ ] 01 Datasets sorted and zipped locally.
- [ ] Zips uploaded to Google Drive.
- [ ] Classifiers trained on Colab (ONNX files generated).
- [ ] YOLO trained on Colab (ONNX file generated).
- [ ] Weights placed in `/backend/weights/`. 
- [ ] Servers restarted and tested via UI.
