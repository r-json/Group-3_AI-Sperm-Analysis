# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation structure
- Professional GitHub repository setup
- Contributing guidelines and templates

## [1.0.0] - 2024-12-30

### Added
- Initial release of AI-assisted sperm morphology classification system
- Deep learning models for HuSHeM and SMIDS datasets
- Modern PyQt5 GUI application with blue gradient theme
- Transfer learning implementation with Xception, MobileNet, and GoogleNet
- Model performance evaluation and validation framework
- Diagnostic tools for system troubleshooting
- Comprehensive project documentation

### Features
- **Multi-Dataset Support**: HuSHeM (4 classes) and SMIDS (3 classes)
- **High Accuracy Models**: 64.4% on HuSHeM, 81.5% on SMIDS
- **Modern GUI Interface**: User-friendly drag-and-drop functionality
- **Real-time Analysis**: Instant classification with confidence scores
- **Model Selection**: Support for multiple neural network architectures
- **Robust Error Handling**: Comprehensive exception management
- **Professional Documentation**: Academic-quality documentation and guides

### Technical Implementation
- TensorFlow 2.x with Keras for deep learning
- PyQt5 for modern desktop GUI
- Transfer learning from ImageNet pre-trained models
- 5-fold cross-validation for model evaluation
- Automated image preprocessing and augmentation
- Model ensemble capabilities for improved accuracy

### Performance Metrics
- **HuSHeM Dataset**:
  - Xception: 64.4% accuracy
  - MobileNet: 61.2% accuracy
  - GoogleNet: 58.9% accuracy
- **SMIDS Dataset** (Transfer Learning):
  - Xception: 81.5% accuracy
  - MobileNet: 78.3% accuracy
  - GoogleNet: 75.1% accuracy

### Dataset Classifications
- **HuSHeM**: Normal, Tapered, Pyriform, Amorphous sperm morphologies
- **SMIDS**: Acrosome Abnormality, Boya, Normal Sperm classifications

### Project Structure
```
├── Finish.ipynb                     # Main training notebook
├── Interface/
│   ├── Application_Modern.py        # Modern PyQt5 GUI
│   ├── modules.py                   # Core model functionality
│   ├── diagnose.py                  # Diagnostic utilities
│   └── img/                         # UI assets and images
├── HuSHeM-20250630T085035Z-1-001/   # HuSHeM dataset and models
├── SMIDS-20250630T085408Z-1-001/    # SMIDS dataset
├── requirements.txt                 # Python dependencies
├── README.md                        # Comprehensive project documentation
├── LICENSE                          # MIT license
└── CONTRIBUTING.md                  # Contribution guidelines
```

### Academic Context
- Developed by Group 3, BSCS 3-1N
- Polytechnic University of the Philippines
- Academic Year 2024-2025
- Focus on biomedical image classification and clinical applications

### Known Issues
- None at this release

### Dependencies
- Python 3.8+
- TensorFlow 2.x
- PyQt5
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Version History

### [1.0.0] - 2024-12-30
- Initial stable release
- Complete feature set implementation
- Professional documentation and repository setup

---

**Note**: This project follows semantic versioning. Version numbers indicate:
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes
