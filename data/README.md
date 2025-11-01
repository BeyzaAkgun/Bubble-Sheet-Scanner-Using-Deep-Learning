# Dataset Information

## 📊 Dataset Overview

This project uses a comprehensive bubble sheet dataset for Optical Mark Recognition (OMR) research.

### Dataset Specifications
- **Total Scanned Exams**: 5,042 high-resolution PNG images
- **Questions per Exam**: Up to 47 questions
- **Answer Formats**: Horizontal and vertical alignments
- **Annotations**: JSON files with bounding boxes and ground truth answers

### 📁 Dataset Structure
dataset/
├── scanned_images/ # Original exam scans
├── annotations/ # JSON annotation files
│ └── dataset_updated.json
├── masks/ # Segmentation masks
└── cropped_regions/ # Preprocessed answer areas


### 🔒 Dataset Access

Due to privacy concerns and large file sizes, the actual dataset files are not included in this repository. The dataset was provided by Prof. Ozgur Ozdemir for academic research purposes.

To use this code with your own data:
1. Place your scanned images in `data/raw/`
2. Prepare JSON annotations in the format described below
3. Update paths in configuration files

### 📋 Annotation Format

Example JSON entry:
```json
{
  "path": "001/1234567891019.png",
  "values": ["0","1","2","3","4","5","6","7","8","9"],
  "alignment": "horizontal",
  "nforms": 1,
  "answer": [["641432160822"]],
  "answer_area": [141, 1268, 1514, 2062]
}


