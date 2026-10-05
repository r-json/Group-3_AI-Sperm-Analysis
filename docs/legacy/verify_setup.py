#!/usr/bin/env python3
"""
GitHub Repository Setup Verification Script
AI-Assisted Diagnosis of Male Fertility Project

This script verifies that all necessary files are present and properly configured
for GitHub repository setup and deployment.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description=""):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {filepath} ({size:,} bytes) - {description}")
        return True
    else:
        print(f"❌ {filepath} - MISSING - {description}")
        return False

def check_directory_exists(dirpath, description=""):
    """Check if a directory exists and report status."""
    if os.path.isdir(dirpath):
        file_count = len(list(Path(dirpath).rglob('*')))
        print(f"✅ {dirpath}/ ({file_count} files) - {description}")
        return True
    else:
        print(f"❌ {dirpath}/ - MISSING - {description}")
        return False

def main():
    """Main verification function."""
    print("🔍 GitHub Repository Setup Verification")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    all_checks_passed = True
    
    # Essential documentation files
    print("\n📚 Documentation Files:")
    files_to_check = [
        ("README.md", "Comprehensive project documentation"),
        ("LICENSE", "MIT license for open source"),
        ("CONTRIBUTING.md", "Contribution guidelines"),
        ("CHANGELOG.md", "Version history and changes"),
        ("CITATION.md", "Academic citation information"),
        ("PROJECT_SUMMARY.md", "Executive project summary"),
        ("GITHUB_SETUP.md", "Repository setup instructions"),
        (".gitignore", "Git ignore configuration"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    for filename, description in files_to_check:
        if not check_file_exists(filename, description):
            all_checks_passed = False
    
    # Core application files
    print("\n🖥️ Application Files:")
    app_files = [
        ("Finish.ipynb", "Main training notebook"),
        ("Interface/Application_Modern.py", "Modern GUI application"),
        ("Interface/modules.py", "Core model functionality"),
        ("Interface/diagnose.py", "Diagnostic utilities"),
    ]
    
    for filename, description in app_files:
        if not check_file_exists(filename, description):
            all_checks_passed = False
    
    # Dataset directories
    print("\n📊 Dataset Directories:")
    dataset_dirs = [
        ("HuSHeM-20250630T085035Z-1-001", "HuSHeM dataset and models"),
        ("SMIDS-20250630T085408Z-1-001", "SMIDS dataset"),
        ("Interface/img", "UI assets and images"),
    ]
    
    for dirname, description in dataset_dirs:
        if not check_directory_exists(dirname, description):
            all_checks_passed = False
    
    # Check requirements.txt content
    print("\n📦 Dependency Verification:")
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().strip()
            if requirements:
                req_lines = [line.strip() for line in requirements.split('\n') if line.strip()]
                print(f"✅ requirements.txt contains {len(req_lines)} dependencies")
                print("   Key dependencies found:")
                key_deps = ['tensorflow', 'PyQt5', 'opencv', 'numpy', 'matplotlib']
                for dep in key_deps:
                    if any(dep.lower() in line.lower() for line in req_lines):
                        print(f"   ✅ {dep}")
                    else:
                        print(f"   ⚠️ {dep} - not explicitly listed")
            else:
                print("❌ requirements.txt is empty")
                all_checks_passed = False
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        all_checks_passed = False
    
    # Check README.md content
    print("\n📖 README Verification:")
    try:
        with open("README.md", "r", encoding='utf-8') as f:
            readme_content = f.read()
            readme_sections = [
                "# 🔬 AI-Assisted Diagnosis",
                "## 🎯 Purpose and Motivation",
                "## 🏗️ System Architecture",
                "## 🚀 Installation and Usage",
                "## 📊 Performance Analysis",
                "## 🎓 Academic Context"
            ]
            
            for section in readme_sections:
                if section in readme_content:
                    print(f"✅ {section}")
                else:
                    print(f"❌ Missing section: {section}")
                    all_checks_passed = False
    except Exception as e:
        print(f"❌ Error reading README.md: {e}")
        all_checks_passed = False
    
    # Check for potential issues
    print("\n⚠️ Potential Issues Check:")
    
    # Check for large files that might need Git LFS
    large_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                if size > 100 * 1024 * 1024:  # 100MB
                    large_files.append((filepath, size))
            except:
                pass
    
    if large_files:
        print("⚠️ Large files detected (>100MB) - Consider Git LFS:")
        for filepath, size in large_files:
            print(f"   📁 {filepath} ({size / (1024*1024):.1f} MB)")
    else:
        print("✅ No files >100MB - Git standard handling sufficient")
    
    # Check for sensitive information
    sensitive_patterns = ['.env', 'password', 'secret', 'api_key', 'token']
    print("\n🔒 Security Check:")
    sensitive_found = False
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(pattern in file.lower() for pattern in sensitive_patterns):
                print(f"⚠️ Potential sensitive file: {os.path.join(root, file)}")
                sensitive_found = True
    
    if not sensitive_found:
        print("✅ No obvious sensitive files detected")
    
    # Final summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Repository is ready for GitHub commit")
        print("\nNext steps:")
        print("1. Install Git if not already installed")
        print("2. Follow instructions in GITHUB_SETUP.md")
        print("3. Commit to https://github.com/r-json/Project-CUM")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("⚠️ Please fix the issues above before committing")
        print("\nRecommendations:")
        print("1. Ensure all required files are present")
        print("2. Verify file contents are complete")
        print("3. Re-run this script to verify fixes")
    
    print("\n📞 For support, see CONTRIBUTING.md or contact the team")
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
