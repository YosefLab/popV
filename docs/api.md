# API

Import popV as:

```python
import popv
```

```{eval-rst}
.. currentmodule:: popv

```

## Preprocessing

For convenience we provide a class that processes query and reference dataset and creates a concatenated dataset.
All relevant entries for annotation are stored in the uns object of the returned AnnData.

Import as:

```python
from popv.preprocessing import Process_Query
```

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   preprocessing.Process_Query
```

## Annotation pipeline

This is the core functionality of popV that performs annotation and consensus voting.
AlgorithmsNT is a named tuple that summarizes all existing cell-type predictors.

Import as:

```python
from popv.annotation import annotate_data
```

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   annotation.AlgorithmsNT
   annotation.annotate_data
```

## Visualization

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   visualization.agreement_score_bar_plot
   visualization.prediction_score_bar_plot
   visualization.celltype_ratio_bar_plot
   visualization.make_agreement_plots
```

## Hub

Pretrained models are stored on
[HuggingFace](https://huggingface.co/popV) and can be downloaded by using
[pull_from_huggingface_hub](popv.hub.HubModel.pull_from_huggingface_hub)
that returns a HubModel class.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   hub.HubMetadata
   hub.HubModel
   hub.HubModelCardHelper
   hub.create_criticism_report
```

## Algorithms

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   algorithms.BaseAlgorithm
   algorithms.KNN_SCVI
   algorithms.SCANVI_POPV
   algorithms.KNN_BBKNN
   algorithms.KNN_HARMONY
   algorithms.Support_Vector
   algorithms.Random_Forest
   algorithms.XGboost
   algorithms.ONCLASS
   algorithms.KNN_SCANORAMA
   algorithms.CELLTYPIST
```
