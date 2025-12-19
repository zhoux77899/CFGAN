<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs\source\cfgan-logo-dark.svg">
    <img alt="cfgan" src="docs\source\cfgan-logo-light.svg" width=50%>
  </picture>
</p>

<h3 align="center">
CFGAN: Cross-Filter Generative Adversarial Network Framework for Microscopy Image Processing
</h3>

---

# Overview
CFGAN is a Python library for advanced image processing tasks leveraging Conditional Generative Adversarial Networks (cGANs). The library is designed for applications such as image denoising, super-resolution reconstruction, and localization tasks in microscopy and other imaging domains.

# Features
- **U-Net Generator**: Advanced encoder-decoder architecture with skip connections for high-quality image generation
- **CNN Discriminator**: Deep convolutional discriminator for effective real/fake classification
- **Comprehensive Training Pipeline**: Complete framework for model training, evaluation, and monitoring
- **Image Processing Toolkit**: Utilities for image manipulation, analysis, and visualization
- **Flexible Configuration System**: YAML-based configuration for model parameters and training settings

# Installation

## Prerequisites
- Python 3.10 or higher
- PyTorch 2.5.1 or higher
- NumPy 2.0.0 or higher
- Other dependencies listed in `requirements.txt`

## From Source
1. Clone the repository:
   ```bash
   git clone https://github.com/zhoux77899/CFGAN.git
   cd CFGAN
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# Contributing
Contributions to CFGAN are welcome! Here are some ways you can contribute:

1. **Report Issues**: Submit bug reports and feature requests through the GitHub issue tracker.
2. **Submit Pull Requests**: Contribute code fixes, improvements, or new features.
3. **Improve Documentation**: Help enhance the project documentation and examples.

Before contributing, please ensure that your code adheres to the project's coding standards and passes all tests.

# License
This project is licensed under the [MIT License](LICENSE).

# Contact
For questions or inquiries about the project, please contact:

- Author: zhoux77899
- GitHub: https://github.com/zhoux77899/CFGAN