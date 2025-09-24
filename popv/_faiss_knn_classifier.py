import os

import faiss
import numpy as np
import pandas as pd

from popv import settings


class FAISSKNNProba:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.index = None
        if settings.cuml:
            self.res = faiss.StandardGpuResources()
        else:
            self.res = None

    def fit(self, X, labels):
        X = X.astype("float32")
        self.labels = labels
        d = X.shape[1]

        cpu_index = faiss.IndexFlatL2(d)

        if settings.cuml:
            gpu_index = faiss.index_cpu_to_gpu(self.res, settings.device, cpu_index)
            gpu_index.add(X)
            self.index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            cpu_index.add(X)
            self.index = cpu_index

        return self

    def query(self, X, n_neighbors):
        X = X.astype("float32")
        if settings.cuml:
            index = faiss.index_cpu_to_gpu(self.res, settings.device, self.index)
        else:
            index = self.index
        _, I = index.search(X, n_neighbors)
        return I

    def predict(self, X, classes):
        X = X.astype("float32")
        if settings.cuml:
            index = faiss.index_cpu_to_gpu(self.res, settings.device, self.index)
        else:
            index = self.index
        _, I = index.search(X, self.n_neighbors)
        preds = classes[np.array([np.bincount(self.labels[i], minlength=len(classes)).argmax() for i in I])]
        return preds

    def predict_proba(self, X, classes):
        X = X.astype("float32")
        if settings.cuml:
            index = faiss.index_cpu_to_gpu(self.res, settings.device, self.index)
        else:
            index = self.index
        _, I = index.search(X, self.n_neighbors)
        probas = []
        for neighbors in I:
            counts = np.bincount(self.labels[neighbors], minlength=len(classes))
            probas.append(counts / counts.sum())
        return np.array(probas)

    def save(self, path_prefix):
        """
        Save FAISS index and metadata (labels + classes) to disk.

        Parameters
        ----------
        path_prefix : str
            Path prefix, e.g. "models/faiss_knn"
        """
        faiss.write_index(self.index, f"{path_prefix}.index")

    @classmethod
    def load(cls, path_prefix, index, n_neighbors=5):
        """
        Load FAISS index and metadata from disk.

        Parameters
        ----------
        path_prefix : str
            Path prefix used in save()
        n_neighbors : int
            Number of neighbors to use
        """
        obj = cls(n_neighbors=n_neighbors)
        obj.index = faiss.read_index(os.path.join(path_prefix, f"{index}.index"))
        labels = pd.read_csv(os.path.join(path_prefix, "ref_labels.csv"), index_col=0)
        obj.labels = labels.iloc[:, 0].to_numpy()
        return obj
