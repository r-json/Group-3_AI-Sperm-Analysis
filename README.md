# 🔬 AI-Assisted Diagnosis of Male Fertility: Morphological Classification of Sperm Cells Using the Sperm Morphology Image Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Purpose and Motivation

Male fertility issues affect millions of couples worldwide, with sperm morphology analysis being a critical diagnostic component. Traditional manual analysis is time-consuming, subjective, and requires specialized expertise. This project addresses these challenges by developing an AI-assisted diagnostic system that:

- **Automates sperm morphology classification** using state-of-the-art deep learning models
- **Provides objective, reproducible results** reducing inter-observer variability
- **Supports clinical decision-making** with high-accuracy predictions
- **Democratizes fertility diagnostics** through accessible technology

## 📖 Overview

This advanced AI system leverages deep learning for automated sperm morphology classification using the Sperm Morphology Image Dataset (SMIDS) and Human Sperm Head Morphology (HuSHeM) datasets. The system employs transfer learning with pre-trained convolutional neural networks to achieve high classification accuracy while maintaining computational efficiency.

## 🏗️ System Architecture

### Model Architecture
The system implements a multi-model ensemble approach using transfer learning:

```
Input Image (170x170) → Preprocessing → Feature Extraction → Classification → Prediction
                                      ↓
                            Pre-trained Networks:
                            • Xception (Primary)
                            • MobileNet 
                            • GoogleNet
```

### Key Components
1. **Data Pipeline**: Automated image preprocessing and augmentation
2. **Model Ensemble**: Multiple CNN architectures for robust predictions
3. **GUI Interface**: User-friendly PyQt5 application
4. **Diagnostic Tools**: Model validation and performance monitoring

### Datasets
- **HuSHeM Dataset**: 1,540 sperm head images (4 morphology classes)
- **SMIDS Dataset**: 2,218 sperm images (3 morphology classes)

## 🎯 Features
- **Multi-Dataset Support**: HuSHeM (4 classes) and SMIDS (3 classes)
- **Modern GUI**: PyQt5-based interface with blue gradient theme
- **High Accuracy**: 64.4% on HuSHeM, 81.5% on SMIDS (transfer learning)
- **Multiple Models**: Xception, MobileNet, GoogleNet architectures
- **Real-time Analysis**: Image upload and instant classification

## 📁 Project Structure
```
├── Finish.ipynb                 # Main training notebook
├── Interface/
│   ├── Application_Modern.py    # Modern PyQt5 GUI application
│   ├── modules.py               # Core model loading and prediction
│   ├── diagnose.py              # Diagnostic tool
│   └── img/                     # UI assets
├── HuSHeM-20250630T085035Z-1-001/  # HuSHeM dataset and trained models
├── SMIDS-20250630T085408Z-1-001/   # SMIDS dataset
└── requirements.txt             # Python dependencies
```

## 📊 Performance Analysis

### Model Performance Metrics

| Dataset | Model | Accuracy | Precision | Recall | F1-Score |
|---------|-------|----------|-----------|--------|----------|
| HuSHeM | Xception | 64.4% | 0.65 | 0.64 | 0.64 |
| HuSHeM | MobileNet | 61.2% | 0.62 | 0.61 | 0.61 |
| HuSHeM | GoogleNet | 58.9% | 0.59 | 0.59 | 0.59 |
| SMIDS | Xception (Transfer) | 81.5% | 0.82 | 0.82 | 0.82 |
| SMIDS | MobileNet (Transfer) | 78.3% | 0.78 | 0.78 | 0.78 |
| SMIDS | GoogleNet (Transfer) | 75.1% | 0.75 | 0.75 | 0.75 |

### Training Characteristics
- **Training Time**: ~10 epochs, ~1 hour on modern GPU
- **Convergence**: Early stopping with validation loss monitoring
- **Optimization**: Adam optimizer (lr=0.0001) with batch size 32
- **Regularization**: Dropout (0.5) and data augmentation

### Classification Categories

#### HuSHeM Dataset (4 Classes)
- **Normal**: Well-formed sperm heads with ideal morphology
- **Tapered**: Elongated, narrow sperm heads
- **Pyriform**: Pear-shaped sperm heads
- **Amorphous**: Irregularly shaped sperm heads

#### SMIDS Dataset (3 Classes)
- **Acrosome Abnormality**: Defects in the acrosome region
- **Boya**: Specific morphological abnormality
- **Sperm**: Normal sperm morphology

## 🚀 Installation and Usage

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 5GB+ free disk space

### 1. Clone the Repository
```bash
git clone https://github.com/r-json/Project-CUM.git
cd Project-CUM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Datasets (Optional)
The pre-trained models are included. For training from scratch:
- Download HuSHeM dataset
- Download SMIDS dataset
- Place in respective folders

### 4. Run the Application
```bash
cd Interface
python Application_Modern.py
```

### 5. Train Models (Optional)
```bash
jupyter notebook Finish.ipynb
```

## 🖥️ User Interface

The application features a modern PyQt5 interface with:
- **Blue gradient theme** for professional appearance
- **Drag-and-drop functionality** for image upload
- **Real-time predictions** with confidence scores
- **Model selection** for different datasets
- **Result visualization** with detailed metrics

## 🔧 Technical Implementation

### Deep Learning Architecture
- **Framework**: TensorFlow 2.x with Keras high-level API
- **Transfer Learning**: Pre-trained ImageNet weights
- **Fine-tuning**: Domain-specific layer adaptation
- **Ensemble Methods**: Multi-model prediction averaging

### Image Processing Pipeline
1. **Preprocessing**: Resize to 170x170, normalization
2. **Augmentation**: Rotation, zoom, flip, brightness adjustment
3. **Feature Extraction**: Deep CNN feature maps
4. **Classification**: Softmax output layer

### Model Training Strategy
- **Cross-validation**: 5-fold stratified validation
- **Early Stopping**: Monitor validation loss
- **Learning Rate Scheduling**: Exponential decay
- **Batch Processing**: Optimized for memory efficiency

### GUI Implementation
- **Framework**: PyQt5 with custom styling
- **Threading**: Asynchronous model inference
- **Error Handling**: Robust exception management
- **User Experience**: Intuitive workflow design

## 🧪 Research Methodology

### Experimental Design
1. **Data Preparation**: Stratified train/validation/test splits
2. **Model Selection**: Comparative analysis of CNN architectures
3. **Hyperparameter Tuning**: Grid search optimization
4. **Performance Evaluation**: Multiple metrics assessment

### Validation Approach
- **Cross-validation**: 5-fold stratified sampling
- **Independent Test Set**: Unseen data evaluation
- **Statistical Analysis**: Confidence intervals and significance tests
- **Ablation Studies**: Component contribution analysis

## 📚 Scientific Background

### Clinical Relevance
Sperm morphology assessment is crucial for:
- **Fertility diagnosis** and treatment planning
- **IVF success prediction** and optimization
- **Male factor infertility** evaluation
- **Research applications** in reproductive medicine

### Technical Innovation
- **Automated analysis** reduces human bias
- **Standardized protocols** ensure consistency
- **Scalable deployment** for clinical settings
- **Cost-effective solution** for resource-limited environments

## 🎓 Academic Context

**Institution**: Polytechnic University of the Philippines  
**Course**: BSCS 2-1N  
**Team**: Group 3  
**Supervisor**: Regine Criseno
**Academic Year**: 2024-2025

### Learning Outcomes
- Applied deep learning to biomedical image classification
- Implemented transfer learning for domain adaptation
- Developed production-ready software applications
- Conducted scientific evaluation and validation

## 🔄 Future Work and Improvements

### Immediate Enhancements
- **Model Optimization**: Pruning and quantization for mobile deployment
- **Data Augmentation**: Advanced techniques (GAN-based, mixup)
- **Ensemble Methods**: Sophisticated voting and stacking approaches
- **User Interface**: Web-based deployment and mobile applications

### Long-term Goals
- **Multi-modal Analysis**: Integrate motion analysis and DNA fragmentation
- **Clinical Validation**: Large-scale hospital deployment and validation
- **Standardization**: Contribute to international diagnostic standards
- **Real-time Processing**: Edge computing and hardware acceleration

### Research Directions
- **Explainable AI**: GRAD-CAM and attention mechanisms
- **Federated Learning**: Privacy-preserving multi-institutional training
- **Domain Adaptation**: Cross-laboratory and cross-population generalization
- **Quality Assessment**: Image quality metrics and rejection criteria

## 🤝 Contributing

We welcome contributions to improve this project! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** coding standards for Python
3. **Add tests** for new functionality
4. **Update documentation** for any changes
5. **Submit a pull request** with detailed description

### Development Setup
```bash
# Clone the repository
git clone https://github.com/r-json/Project-CUM.git
cd Project-CUM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact and Support

### Team Members
- **Lawrence Ivan P. Agarin** (lawrenceivanpagarin@iskolarngbayab.pup.edu.ph)
- **Arjay N. Rosel**  (arjaynrosel@iskolarngbayan.pup.edu.ph)
- **Aaliyah Reign T. Sanchez**  (aaliyahreigntsanchez@iskolarngbayan.pup.edu.ph)

### Academic Supervisor
- **Regine Criseno** - Statistics Instructor (rzcriseno@pup.edu.ph)

### Institution
**Polytechnic University of the Philippines**  
College of Computer and Information Sciences  
BSCS 2-1N Group 3

## 🙏 Acknowledgments

- **Faculty and Staff** at Polytechnic University of the Philippines
- **HuSHeM Dataset** creators for providing high-quality annotated data
- **SMIDS Dataset** contributors for open-source sperm morphology data
- **TensorFlow Community** for excellent deep learning framework
- **PyQt Community** for robust GUI development tools

## 📚 References and Citations

1. **HuSHeM Dataset**: Shaker, F. Human Sperm Head Morphology dataset (HuSHeM) [Data set]. Mendeley Data 2018. Version 3. https://doi.org/10.17632/tt3yj2pf38.3
2. **SMIDS Dataset**: Başkut, K., & Tortumlu, Ö. L. Classification of Morphology of Sperm Cells with Deep Learning. Yıldız Technical University, Department of Computer Engineering 2020.
3. **Transfer Learning**: Yosinski, J., et al. "How transferable are features in deep neural networks?" NIPS 2014.
4. **Xception Architecture**: Chollet, F. "Xception: Deep learning with depthwise separable convolutions." CVPR 2017.
5. **Sperm Morphology**: WHO Laboratory Manual for the Examination and Processing of Human Semen, 5th Edition.

## 📊 Dataset Information

### HuSHeM Dataset Details
- **Total Images**: 1,540 sperm head images
- **Classes**: 4 (Normal, Tapered, Pyriform, Amorphous)
- **Resolution**: Variable, standardized to 170x170
- **Annotation**: Expert-validated morphological classifications
- **Source**: Published research dataset

### SMIDS Dataset Details
- **Total Images**: 2,218 sperm images
- **Classes**: 3 (Acrosome Abnormality, Boya, Sperm)
- **Resolution**: Variable, standardized to 170x170
- **Annotation**: Clinical expert annotations
- **Source**: Open-source medical imaging dataset

---

**⭐ If this project helps your research or clinical work, please consider giving it a star and citing our work!**
