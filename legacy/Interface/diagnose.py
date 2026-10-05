#!/usr/bin/env python3
"""
Diagnostic Tool for Sperm Morphology Application
This will help identify and fix common issues
"""

import os
import sys
import traceback
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        missing_deps.append("tensorflow")
        print("❌ TensorFlow: Not installed")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
        print("❌ NumPy: Not installed")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
        print("❌ OpenCV: Not installed")
    
    try:
        import skimage
        print(f"✅ Scikit-image: {skimage.__version__}")
    except ImportError:
        missing_deps.append("scikit-image")
        print("❌ Scikit-image: Not installed")
    
    try:
        from PyQt5 import QtWidgets
        print("✅ PyQt5: Available")
    except ImportError:
        missing_deps.append("PyQt5")
        print("❌ PyQt5: Not installed")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install them with: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_model_files():
    """Check if model files exist in the expected locations"""
    print("\n🔍 Checking model files...")
    
    base_path = Path("../HuSHeM-20250630T085035Z-1-001/HuSHeM/")
    
    # Check JSON file
    json_file = base_path / "mobil.json"
    if json_file.exists():
        print(f"✅ Model JSON: {json_file}")
    else:
        print(f"❌ Model JSON missing: {json_file}")
        return False
    
    # Check weights file
    weights_file = base_path / "Mobil_Hushem.weights.h5"
    if weights_file.exists():
        size_mb = weights_file.stat().st_size / (1024 * 1024)
        print(f"✅ Model weights: {weights_file} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Model weights missing: {weights_file}")
        return False
    
    print("✅ All model files found!")
    return True

def test_modules_import():
    """Test if modules.py can be imported and used"""
    print("\n🔍 Testing modules.py...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import modules
        print("✅ modules.py imported successfully")
        
        # Test the helper function
        info = modules.get_model_info()
        print(f"✅ Model info function works: {len(info['datasets'])} datasets available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing modules.py: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_basic_model_loading():
    """Test basic model loading without prediction"""
    print("\n🔍 Testing model loading...")
    
    try:
        import modules
        
        # Test if we can at least call the function (even if no image)
        result_text, prediction = modules.test_model("HuSHeM", "Xception", "")
        
        if "Please specify" in result_text:
            print("✅ Model loading function responds correctly to empty input")
            return True
        else:
            print(f"❌ Unexpected response: {result_text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model loading: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def main():
    print("🚀 Sperm Morphology Application Diagnostic Tool")
    print("=" * 50)
    
    all_good = True
    
    # Run all checks
    all_good &= check_dependencies()
    all_good &= check_model_files()
    all_good &= test_modules_import()
    all_good &= test_basic_model_loading()
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! The application should work correctly.")
        print("🚀 You can now run: python Application_Modern.py")
    else:
        print("❌ Some issues were found. Please fix them before running the application.")
    
    print("\n💡 If you're still having issues, please share the specific error message.")

if __name__ == "__main__":
    main()
