from ._bbknn import BBKNN as knn_on_bbknn
from ._celltypist import CELLTYPIST as celltypist
from ._harmony import HARMONY as knn_on_harmony
from ._onclass import ONCLASS as onclass
from ._rf import RF as rf
from ._scanorama import SCANORAMA as knn_on_scanorama
from ._scanvi import SCANVI_POPV as scanvi
from ._scvi import SCVI_POPV as knn_on_scvi
from ._svm import SVM as svm

__all__ = [
    "base_algorithm",
    "celltypist",
    "knn_on_bbknn",
    "knn_on_harmony",
    "knn_on_scanorama",
    "knn_on_scvi",
    "onclass",
    "rf",
    "scanvi",
    "svm",
]
