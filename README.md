# Land classification

In this project I use different deep learning models (both from scratch and transfer learning) to classify Sentinel-2 satellite images into 10 land-use / land-cover classes (forest, river, highway, residential...) using the [EuroSAT](https://github.com/phelber/EuroSAT) dataset. Concretely, the notebook compares three approaches: 

- **CNN trained from scratch**: a small standard model with three convolutional blocks and a linear head (~175 K parameters).
- **ResNet-18 with frozen backbone**: features pre-trained on ImageNet, training only the final adapted layer (~5 K trainable parameters).
- **ResNet-18 with full fine-tuning**: training of the whole model, starting from the pre-trained weights (~11 M parameters).

All three models are trained, validated and tested on exactly the same split of the data. The notebook finishes with a comparison of the three, summarising the obtained accuracy with a 95% confidence interval, number of trainable parameters and training time per epoch. Fine-tuning the pre-trained ResNet-18 is found to give the best accuracy, beating the other models' best results already after its first epoch. Since the time spent per epoch is rougly the same on all three models, the loading of the images is probably the bottleneck of the computations, and thus using the fully fine-tuned ResNet-18 pays off even on time grounds in this setup.


## Structure
- [notebooks/land_classification.ipynb](notebooks/land_classification.ipynb): the notebook going through data preparation, the three models and their comparison.
- [src/utils.py](src/utils.py): training and evaluation functions shared by the three models, imported in the notebook.
- [environment.yml](environment.yml): conda environment with all required packages.


## Setup

The notebook requires Python 3.12 and the following packages:
- pytorch and torchvision
- numpy
- pandas
- matplotlib
- scikit-learn

The easiest way to get them is creating the conda environment from the repository's root folder:

    conda env create -f environment.yml
    conda activate land_classification

The environment installs PyTorch with CUDA 13.0 support, so training runs on an NVIDIA GPU if one is available (it falls back to the CPU otherwise, though much more slowly). For a different CUDA version, change the `--extra-index-url` and the `+cu130` suffixes in `environment.yml` to the ones given at [pytorch.org](https://pytorch.org/get-started/locally/).

The EuroSAT dataset (~90 MB) is downloaded automatically into `data/raw/` the first time the notebook is run, and the best weights of each model are saved into `data/processed/`. Neither folder is tracked by git.


## Data source
- [EuroSAT](https://github.com/phelber/EuroSAT) (RGB version): 27,000 labelled 64 × 64 Sentinel-2 image patches, from Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification* (2019).
- [ImageNet](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) pre-trained ResNet-18 weights, from torchvision.

