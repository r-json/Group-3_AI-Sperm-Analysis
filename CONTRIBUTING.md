# Contributing to AI-Assisted Diagnosis of Male Fertility

Thank you for your interest in contributing to this project! This document outlines the guidelines for contributing to the AI-assisted sperm morphology classification system.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs or request features
- Provide detailed information about the problem or enhancement
- Include system information (OS, Python version, dependencies)
- Add screenshots or error logs when applicable

### Submitting Code Changes
1. **Fork the repository** to your GitHub account
2. **Create a feature branch** from `main`
3. **Make your changes** following the coding standards
4. **Test your changes** thoroughly
5. **Submit a pull request** with a clear description

## 📋 Development Guidelines

### Code Standards
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Keep functions focused and modular

### Testing Requirements
- Write unit tests for new functionality
- Ensure all existing tests pass
- Achieve minimum 80% code coverage
- Test on multiple platforms when possible

### Documentation
- Update README.md for significant changes
- Add inline comments for complex logic
- Document new features in appropriate sections
- Keep documentation current and accurate

## 🔬 Research Contributions

### Dataset Improvements
- New sperm morphology datasets
- Enhanced annotation quality
- Cross-validation with clinical experts
- Multi-institutional data collection

### Model Enhancements
- Novel neural network architectures
- Improved transfer learning strategies
- Ensemble method optimizations
- Performance benchmarking

### Clinical Applications
- Real-world deployment studies
- User interface improvements
- Clinical workflow integration
- Validation studies

## 🏗️ Development Setup

### Environment Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Project-CUM.git
cd Project-CUM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Development Tools
```bash
# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy .

# Testing
pytest tests/ --cov=.
```

## 📝 Pull Request Process

1. **Update documentation** for any new features
2. **Add or update tests** as needed
3. **Ensure CI/CD passes** all checks
4. **Get approval** from project maintainers
5. **Squash commits** before merging

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes generate no new warnings
```

## 🎯 Priority Areas

### High Priority
- Performance optimization
- Model accuracy improvements
- User interface enhancements
- Bug fixes and stability

### Medium Priority
- New dataset integration
- Additional model architectures
- Advanced visualization features
- Mobile/web deployment

### Low Priority
- Code refactoring
- Documentation improvements
- Example tutorials
- Community features

## 🔒 Security Guidelines

- Do not commit sensitive information
- Use environment variables for configurations
- Follow secure coding practices
- Report security vulnerabilities privately

## 📞 Getting Help

- **GitHub Discussions**: General questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Email**: Direct contact for sensitive matters
- **Documentation**: Comprehensive guides and tutorials

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributor section
- Release notes for significant contributions
- Academic publications (when applicable)
- Project presentations and demos

Thank you for helping make this project better! 🙏
