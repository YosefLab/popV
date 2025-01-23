# popV

[![Stars](https://img.shields.io/github/stars/yoseflab/popv?logo=GitHub&color=yellow)](https://github.com/YosefLab/popv/stargazers)
[![PyPI](https://img.shields.io/pypi/v/popv.svg)](https://pypi.org/project/popv)
[![PopV](https://github.com/YosefLab/PopV/actions/workflows/test.yml/badge.svg)](https://github.com/YosefLab/PopV/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/YosefLab/popv/branch/main/graph/badge.svg?token=KuSsL5q3l7)](https://codecov.io/gh/YosefLab/popv)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Downloads](https://pepy.tech/badge/popv)](https://pepy.tech/project/popv)

PopV uses popular vote of a variety of cell-type transfer tools to classify
cell-types in a query dataset based on a test dataset. Using this variety of
algorithms, we compute the agreement between those algorithms and use this
agreement to predict which cell-types are with a high likelihood the same
cell-types observed in the reference.

## Algorithms

Currently implemented algorithms are:

- K-nearest neighbor classification after dataset integration with
    [BBKNN](https://github.com/Teichlab/bbknn)
- K-nearest neighbor classification after dataset integration with
    [SCANORAMA](https://github.com/brianhie/scanorama)
- K-nearest neighbor classification after dataset integration with
    [scVI](https://github.com/scverse/scvi-tools)
- K-nearest neighbor classification after dataset integration with
    [Harmony](https://github.com/lilab-bcb/harmony-pytorch)
- Random forest classification
- Support vector machine classification
- XGboost classification
- [OnClass](https://github.com/wangshenguiuc/OnClass) cell type classification
- [scANVI](https://github.com/scverse/scvi-tools) label transfer
- [Celltypist](https://www.celltypist.org) cell type classification

All algorithms are implemented as a class in
[popv/algorithms](https://github.com/YosefLab/popV/tree/main/popv/algorithms).

New classifiers should inherit from
[BaseAlgorithm](https://github.com/YosefLab/popV/tree/main/popv/algorithms/_base_algorithm.py). Adding a new class with the
methods defined in this class and adding it to [AlgorithmsNT]([popV/](https://github.com/YosefLab/popV/tree/main/popv/annotation.py)
will tell PopV to include this class into its
classifiers and will use the new classifier as another expert.

All algorithms that allow for pre-training are pre-trained. This excludes by
design BBKNN, Harmony and SCANORAMA as all construct a new embedding space.
To provide pretrained methods for BBKNN and Harmony, we use a nearest-neighbor
index in PCA space and position query cells at the average position of the 5
nearest neighbors.

Pretrained models are stored on
[HuggingFace](https://huggingface.co/popV) and can be downloaded by using
[pull_from_huggingface_hub](https://github.com/YosefLab/popV/tree/main/popv/hub/_model.py)
that returns a class and can annotate
query data by calling the [annotate_data](https://github.com/YosefLab/popV/tree/main/popv/hub/_model.py)
method of that class.

All input parameters are defined during initial call to
\[Process_Query\]([popv.preprocessing.Process_Query](https://github.com/YosefLab/popV/tree/main/popv/preprocessing.py) and are stored in the uns
field of the generated AnnData object. PopV has three levels of prediction
complexities:

- **retrain**: Will train all classifiers from scratch. For 50k cells, this
    takes up to an hour of computing time using a GPU.
- **inference**: Uses pretrained classifiers to annotate query and reference
    cells and construct a joint embedding using all integration methods. For 50k
    cells, this takes up to half an hour of GPU time.
- **fast**: Uses only methods with pretrained classifiers to annotate only
    query cells. For 50k cells, this takes 5 minutes without a GPU (without UMAP
    embedding).

A user-defined selection of classification algorithms can be defined when
calling \[annotate_data\]([popv.annotation.annotate_data](https://github.com/YosefLab/popV/tree/main/popv/annotation.py). Additionally, advanced users
can define non-standard parameters for the integration methods and classifiers.

## Output

PopV will output a cell-type classification for each of the used classifiers,
as well as the majority vote across all classifiers. Additionally, PopV uses
the ontology to go through the full ontology descendants for the OnClass
prediction (disabled in fast mode). This method will be further described when
PopV is published. PopV also outputs a score that counts the number of
classifiers agreeing on the PopV prediction. This can be seen as the certainty
that the current prediction is correct for every single cell in the query data.

We found that disagreement of a single expert is still highly reliable, while
disagreement of more than two classifiers signifies less reliable results. The
aim of PopV is not to fully annotate a dataset but to highlight cells that may
require further manual annotation. PopV also outputs UMAP embeddings of all
integrated latent spaces if `_settings.compute_embedding == True` and computes
certainties for every used classifier if `_settings.return_probabilities == True`.

## Installation

We suggest using a package manager like `conda` or `mamba` to install the
package. OnClass files for annotation based on Tabula sapiens are deposited in
`popv/resources/ontology`. We use [Cell Ontology](https://obofoundry.org/ontology/cl.html)
as an ontology throughout our experiments. PopV will automatically look for the
ontology in this folder. If you want to provide your user-edited ontology,
`popv/add_celltypes_ontology.ipynb` demonstrates how to generate the Natural
Language Model used in OnClass for this user-defined ontology.

```bash
conda create -n yourenv python=3.12
conda activate yourenv
pip install git+https://github.com/YosefLab/popV
```

## Example notebook

We provide an example notebook in Google Colab:

- [Tutorial demonstrating use of Tabula sapiens as a reference](https://github.com/YosefLab/popV/tree/main/docs/tutorials/notebooks/tabula_sapiens_tutorial.ipynb)

This notebook will guide you through annotating a dataset based on the annotated
[Tabula sapiens reference](https://tabula-sapiens-portal.ds.czbiohub.org) and
demonstrates how to run annotation on your own query dataset. This notebook
requires that all cells are
annotated based on a cell ontology. We strongly encourage the use of a
common cell ontology,
see also [Osumi-Sutherland et al](https://www.nature.com/articles/s41556-021-00787-7).
Using a cell ontology is a requirement to run OnClass as a prediction algorithm.
Setting ontology
to false, will disable this step and allows running popV without using a cell ontology.
