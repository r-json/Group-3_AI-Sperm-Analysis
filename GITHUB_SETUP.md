# GitHub Repository Setup Guide

This guide will help you commit all project files to the GitHub repository at https://github.com/r-json/Project-CUM.

## Prerequisites

1. **Install Git** on your system:
   - Windows: Download from https://git-scm.com/download/windows
   - Or install via Chocolatey: `choco install git`
   - Or install via winget: `winget install Git.Git`

2. **Configure Git** (first time setup):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **GitHub Account**: Ensure you have access to https://github.com/r-json/Project-CUM

## Step-by-Step Repository Setup

### 1. Initialize Local Repository
```bash
cd "C:\Users\rosel\Downloads\Sperm Project_Group 3"
git init

# Set up Git LFS for large model files (REQUIRED)
git lfs install
git lfs track "*.h5"
git lfs track "*.weights.h5" 
git lfs track "*.keras"
git add .gitattributes
```

### 2. Connect to GitHub Repository
```bash
git remote add origin https://github.com/r-json/Project-CUM.git
```

### 3. Add All Project Files
```bash
# Add all files (Git LFS will handle large files automatically)
git add .
```

### 4. Create Initial Commit
```bash
git commit -m "Initial commit: AI-assisted sperm morphology classification system

- Complete deep learning pipeline for HuSHeM and SMIDS datasets
- Modern PyQt5 GUI application with blue gradient theme
- Transfer learning implementation with 64.4% and 81.5% accuracy
- Comprehensive documentation and professional repository structure
- Academic project by Group 3, BSCS 3-1N, PUP"
```

### 5. Push to GitHub
```bash
git branch -M main
git push -u origin main
```

## Alternative: Upload via GitHub Web Interface

If you prefer using the GitHub web interface:

1. **Go to**: https://github.com/r-json/Project-CUM
2. **Click**: "Upload files" button
3. **Drag and drop** all project folders and files
4. **Add commit message**: "Initial commit: Complete AI sperm morphology classification system"
5. **Click**: "Commit changes"

## Files to Commit

The following files and folders should be committed:

### Essential Project Files
- `README.md` - Comprehensive project documentation
- `requirements.txt` - Python dependencies
- `LICENSE` - MIT license
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history
- `CITATION.md` - Academic citation information
- `.gitignore` - Git ignore rules

### Core Application
- `Finish.ipynb` - Main training notebook
- `Interface/` - Complete GUI application folder
  - `Application_Modern.py` - Modern PyQt5 interface
  - `modules.py` - Core model functionality
  - `diagnose.py` - Diagnostic utilities
  - `img/` - UI assets and images

### Datasets and Models
- `HuSHeM-20250630T085035Z-1-001/` - HuSHeM dataset and trained models
- `SMIDS-20250630T085408Z-1-001/` - SMIDS dataset

## Repository Structure After Commit

```
Project-CUM/
├── README.md                           # Comprehensive documentation
├── LICENSE                             # MIT license
├── CONTRIBUTING.md                     # Contribution guidelines
├── CHANGELOG.md                        # Version history
├── CITATION.md                         # Academic citations
├── requirements.txt                    # Dependencies
├── .gitignore                          # Git ignore rules
├── Finish.ipynb                        # Training notebook
├── Interface/                          # GUI application
│   ├── Application_Modern.py           # Main application
│   ├── modules.py                      # Core functionality
│   ├── diagnose.py                     # Diagnostic tools
│   └── img/                            # UI assets
├── HuSHeM-20250630T085035Z-1-001/      # HuSHeM dataset
└── SMIDS-20250630T085408Z-1-001/       # SMIDS dataset
```

## GitHub Repository Enhancements

After committing, consider adding:

### 1. Repository Description
Set repository description to: "AI-assisted diagnosis system for male fertility assessment through automated sperm morphology classification using deep learning"

### 2. Topics/Tags
Add relevant topics:
- `machine-learning`
- `deep-learning`
- `medical-imaging`
- `tensorflow`
- `computer-vision`
- `biomedical`
- `fertility`
- `classification`
- `transfer-learning`
- `pyqt5`

### 3. Repository Settings
- Enable Issues for bug reports
- Enable Discussions for community
- Add repository website (if deploying)
- Set up branch protection rules

### 4. GitHub Pages (Optional)
Enable GitHub Pages for documentation hosting

### 5. README Badges
The README already includes status badges that will work once the repository is set up

## Verification

After committing, verify the repository:

1. **Check all files** are present on GitHub
2. **Review README** renders correctly
3. **Test clone** from another location:
   ```bash
   git clone https://github.com/r-json/Project-CUM.git
   cd Project-CUM
   pip install -r requirements.txt
   cd Interface
   python Application_Modern.py
   ```

## Troubleshooting

### Common Issues
- **Large files**: Use Git LFS for model files > 100MB
- **Permissions**: Ensure you have write access to the repository
- **Network**: Check internet connection for push operations

### Git LFS Setup (if needed for large model files)
```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.json"
git add .gitattributes
git commit -m "Add Git LFS for model files"
```

## Next Steps

After successful repository setup:

1. **Share repository** with team members
2. **Set up CI/CD** for automated testing
3. **Create releases** for version management
4. **Monitor issues** and community feedback
5. **Plan future development** based on user needs

---

**Note**: This guide assumes you have the necessary permissions for the GitHub repository. Contact the repository owner if you encounter access issues.
