# Bubble Sheet Scanner Using Deep Learning

This repository contains the implementation of a novel, template-free Optical Mark Recognition (OMR) pipeline that unifies semantic segmentation and image captioning for end-to-end answer prediction from bubble sheet exam papers. The system addresses limitations of traditional OMR systems by leveraging deep learning to handle diverse layouts, scanning artifacts, and marking inconsistencies.

## Key Features

- **Template-free OMR pipeline** that doesn't require predefined layouts or fiducial markers
- **Hybrid architecture** combining semantic segmentation with image captioning
- **Multiple segmentation backbones** evaluated: U-Net, ResNet-34, and SegFormer
- **Vision-language integration** using EfficientNet-B0 + Transformer for answer generation
- **Comprehensive evaluation** against general-purpose vision-language models (VLMs)
- **Robust preprocessing** handling various scanning conditions and artifacts

## Methodology

### Pipeline Overview
1. **Data Preprocessing**: Grayscale conversion, normalization, and noise reduction
2. **Segmentation**: Answer area localization using semantic segmentation
3. **Bounding Box Extraction & Normalization**: Padding for consistent dimensions
4. **Image Captioning**: Answer prediction via encoder-decoder architecture
5. **Multimodal Evaluation**: Validation using VLMs and error analysis

### Preliminary Works

#### Early Dataset Analysis and Annotation Trials
Due to the scarcity of public OMR datasets, initial experiments were conducted using the Afifi et al. dataset containing 171 unlabeled scanned exam images. Key preliminary activities included:

- **Annotation Process**: Bubble areas were annotated using VGG Image Annotator (VIA), labeling bubbles as 1 and background as 0
- **Annotation Challenges**: Rectangular bounding-box annotations didn't perfectly match irregular bubble shapes but provided a reasonable baseline
- **Dataset Quality Assessment**: Highlighted the importance of annotation granularity and dataset diversity for robust model performance

#### CNN and SegFormer Comparison with Afifi Dataset
Initial segmentation experiments compared two architectures on the Afifi dataset:

| Metric | U-Net (Afifi) | SegFormer (Afifi) |
|--------|---------------|-------------------|
| Accuracy | 0.8155 | 0.9880 |
| Precision | 0.7517 | 0.9750 |
| Recall | 0.4419 | 0.9793 |
| F1-score | 0.5566 | 0.9771 |
| Dice | 0.5566 | 0.9771 |
| IoU | 0.3856 | 0.9553 |

**Key Findings:**
- **U-Net**: Showed better sensitivity to local geometric irregularities but struggled with coarse annotations
- **SegFormer**: Produced masks more consistent with the coarse ground truth, leveraging self-attention for global consistency
- **Dataset Impact**: Demonstrated that dataset quality strongly affects measured performance, motivating the creation of our larger, more diverse dataset

### Segmentation Approaches
Three architectures were evaluated for answer region localization:
- **U-Net**: Convolutional baseline with encoder-decoder structure
- **ResNet-34**: Deeper residual network for improved feature extraction
- **SegFormer**: Transformer-based model with hierarchical attention

### Answer Prediction
- **Architecture**: EfficientNet-B0 encoder + Transformer decoder
- **Task Formulation**: Treated as image captioning rather than classification
- **Output**: Character-level sequence generation for answer strings

## Dataset

**Note:** The actual dataset files are not included in this repository due to privacy concerns and large file sizes. The dataset consists of 5,042 high-resolution scanned exam papers provided by Prof. Ozgur Ozdemir for academic research purposes.

### Dataset Specifications
- Up to 47 questions per paper
- Horizontal and vertical alignment formats
- JSON annotations with bounding boxes and ground truth answers
- Various marking styles and capture conditions

After filtering and preprocessing, the final dataset includes:
- 2,814 usable exam papers
- 5,454 cropped bubble regions
- 80/10/10 train/validation/test split

For dataset access inquiries or to use your own data, please see the [Dataset Documentation](data/DATASET_README.md).

## Results

### Segmentation Performance
| Model | Dice | IoU | Precision | Recall | Accuracy |
|-------|------|-----|-----------|--------|----------|
| SegFormer | 0.9996 | 0.9991 | 0.9994 | 0.9998 | 0.9999 |
| ResNet-34 | 0.9996 | 0.9991 | 0.9995 | 0.9996 | 0.9999 |
| U-Net | 0.9741 | 0.9500 | 0.9610 | 0.9881 | 0.9932 |

### Captioning Performance
- **Token-level accuracy**: 71.85%
- **Sequence exact-match accuracy**: 52.40%
- **Character Error Rate (CER)**: 0.2113
- **BLEU-1**: 0.5031

### VLM Comparison
| Model | Token Acc. | Seq. Acc. | CER |
|-------|------------|------------|-----|
| Our Captioning | 71.85% | 52.40% | 0.21 |
| MiniCPM-V (best) | 28.19% | 14.90% | 0.92 |
| Nanonets-OCR-s | 15.25% | 1.99% | 1.06 |

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/bubble-sheet-scanner.git
cd bubble-sheet-scanner

# Install dependencies
pip install -r requirements.txt



###  Training
python
# Train segmentation model
python train_segmentation.py --model segformer --dataset path/to/dataset

# Train captioning model
python train_captioning.py --encoder efficientnet-b0 --decoder transformer
Inference
python
from pipeline import BubbleSheetScanner

scanner = BubbleSheetScanner()
results = scanner.process_exam("path/to/exam_sheet.png")
print(f"Predicted answers: {results['answers']}")
Project Structure
text
bubble-sheet-scanner/
├── data/
│   ├── raw/           # Raw scanned images
│   ├── processed/     # Preprocessed images and masks
│   └── annotations/   # JSON annotation files
├── preliminary/
│   ├── afifi_dataset/ # Afifi dataset experiments
│   ├── annotation_trials/ # VIA annotation trials
│   └── early_analysis/ # Initial dataset analysis
├── models/
│   ├── segmentation/  # Segmentation model implementations
│   ├── captioning/    # Captioning model implementations
│   └── vlms/          # Vision-language model wrappers
├── utils/
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── visualization.py
├── pipeline.py        # Main pipeline implementation
└── configs/           # Training and model configurations


### Research Reports
This project is documented through comprehensive research reports showing the evolution of our work:

Original Thesis: reports/bubble_sheet_scanner_using_deep_learning_old.pdf

Enhanced Version: reports/bubble_sheet_scanner_using_deep_learning_new.pdf

For detailed comparison between reports, see Reports Documentation.

### Citation
If you use this code in your research, please cite:

bibtex
@unpublished{akgun2025bubblesheet,
  title={Bubble Sheet Scanner Using Deep Learning},
  author={Akgun, Beyza and Kasapcopur, Ilayda},
  note={Bachelor's Thesis, Istanbul Bilgi University},
  year={2025}
}


### Acknowledgments
Dataset provided by Prof. Ozgur Ozdemir

OMR Checker Tool for alignment correction

Hugging Face for pretrained model access

Afifi et al. for the preliminary dataset used in initial experiments