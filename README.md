# FGSM Attacks on Semantic Segmentation

This project demonstrates Fast Gradient Sign Method (FGSM) adversarial attacks against semantic segmentation models, specifically focusing on the Cityscapes dataset and DeepLabV3 model.

## Project Overview

The project includes tools for:
- Generating adversarial examples using multiple FGSM variants
- Visualizing the effects of these attacks on semantic segmentation predictions
- Comparing different attack methods and strengths

## Features

- **Fast FGSM Methods**: Generate adversarial examples without requiring model gradients
  - Structured pattern-based perturbations
  - Edge-based perturbations
  - Frequency domain perturbations
- **Visualization Tools**: Comprehensive visualization of adversarial effects
- **Segmentation Analysis**: Quantitative analysis of how adversarial examples affect predictions

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- NumPy
- Matplotlib
- SciPy
- PIL/Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fgsm-segmentation.git
cd fgsm-segmentation
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the required datasets and models (see "Downloading Resources" section below)

## Files Excluded from Version Control

Due to their large size, the following files are excluded from version control:

- **Model files**: `models/` directory and `.pb` files
- **Datasets**: `data/cityscapes/` directory
- **Results**: All output directories (`results/`, `quick_demo_results/`, etc.)
- **Generated images**: All PNG, JPG, and JPEG files

## Downloading Resources

### DeepLabV3 Model

The scripts will automatically download the DeepLabV3 model when first run, but you can also download it manually:

```bash
mkdir -p models
cd models
wget http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz
tar -xzf deeplabv3_cityscapes_train_2018_02_06.tar.gz
cd ..
```

### Cityscapes Dataset

1. Register and download the dataset from [Cityscapes Website](https://www.cityscapes-dataset.com/)
2. Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`
3. Create the data directory and extract the files:
```bash
mkdir -p data/cityscapes
unzip leftImg8bit_trainvaltest.zip -d data/cityscapes/
unzip gtFine_trainvaltest.zip -d data/cityscapes/
```

## Usage

### Quick Demo

For a quick demonstration of adversarial effects on segmentation:

```bash
python quick_demo.py --image_path data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/bielefeld/bielefeld_000000_021341_leftImg8bit.png --epsilon 0.02 --method structured
```

This will:
1. Load the specified image and the DeepLabV3 model
2. Generate an adversarial example using the structured pattern method
3. Show how the segmentation results are affected
4. Save visualizations to the `quick_demo_results` directory

### Comparing Attack Methods

To compare different adversarial attack methods:

```bash
python fast_fgsm.py --image_path data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/bielefeld/bielefeld_000000_021341_leftImg8bit.png --epsilon 0.02
```

This will:
1. Apply three different FGSM attack methods to the image (structured, edge, and frequency)
2. Save the perturbed images and visualization of the perturbations
3. Output timing information for each method

### Detailed Example Analysis

For a more comprehensive analysis with different epsilon values:

```bash
python view_single_example.py --image_path data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/bielefeld/bielefeld_000000_021341_leftImg8bit.png
```

This will:
1. Generate adversarial examples with multiple epsilon values
2. Show detailed comparisons of how different perturbation strengths affect the segmentation
3. Visualize the changes in predicted segmentation maps

## Available Attack Methods

- `structured`: Combined patterns specifically designed for semantic segmentation
- `edge`: Perturbations based on image edges
- `frequency`: Perturbations in the frequency domain

## Parameters

- `--image_path`: Path to the input image
- `--epsilon`: Perturbation strength (default: 0.02)
- `--method`: Attack method (default: structured)
- `--output_dir`: Directory to save results

## Project Structure

- `adversarial.py`: Numerical approximation of FGSM
- `fast_fgsm.py`: Fast FGSM implementations
- `model_utils.py`: DeepLabV3 model utilities
- `cityscapes_utils.py`: Cityscapes dataset utilities
- `view_single_example.py`: Detailed visualization of adversarial examples
- `quick_demo.py`: Quick demonstration of adversarial effects
- `example.py`: Example script with comparison functionality

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@misc{fgsm-segmentation,
  author = {Your Name},
  title = {FGSM Attacks on Semantic Segmentation},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/fgsm-segmentation}}
}
```

## Acknowledgments

- The DeepLabV3 model implementation is based on TensorFlow Models
- Cityscapes dataset from [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- FGSM method based on [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)