from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from popv import settings

if settings.cuml:
    import rapids_singlecell as rsc
from popv.algorithms._base_algorithm import BaseAlgorithm


class KNN_SCANORAMA(BaseAlgorithm):
    """
    Class to compute KNN classifier after Scanorama integration.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information. Default is "_batch_annotation".
    labels_key
        Key in obs field of adata for cell-type information. Default is "_labels_annotation".
    result_key
        Key in obs in which celltype annotation results are stored.
        Default is "popv_knn_scanorama_prediction".
    embedding_key
        Key in obsm in which embedding of integrated data is stored.
        Default is "X_pca_scanorama_popv".
    umap_key
        Key in obsm in which UMAP embedding of integrated data is stored.
        Default is "X_umap_scanorama_popv".
    method_kwargs
        Additional parameters for SCANORAMA. Options at :func:`scanpy.external.pp.scanorama_integrate`.
    classifier_kwargs
        Dictionary to supply non-default values for KNN classifier.
        See :class:`sklearn.neighbors.KNeighborsClassifier`.
        Default is {"weights": "uniform", "n_neighbors": 15}.
    embedding_kwargs
        Dictionary to supply non-default values for UMAP embedding.
        See :func:`scanpy.tl.umap`.
        Default is {"min_dist": 0.1}.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_knn_scanorama_prediction",
        embedding_key: str | None = "X_pca_scanorama_popv",
        umap_key: str | None = "X_umap_scanorama_popv",
        method_kwargs: dict | None = None,
        classifier_kwargs: dict | None = None,
        embedding_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
            umap_key=umap_key,
        )

        if embedding_kwargs is None:
            embedding_kwargs = {}
        if classifier_kwargs is None:
            classifier_kwargs = {}
        if method_kwargs is None:
            method_kwargs = {}

        self.method_kwargs = {}
        if method_kwargs is not None:
            self.method_kwargs.update(method_kwargs)

        self.classifier_kwargs = {"weights": "uniform", "n_neighbors": 15}
        if classifier_kwargs is not None:
            self.classifier_kwargs.update(classifier_kwargs)

        self.embedding_kwargs = {
            "min_dist": 0.1,
        }
        self.embedding_kwargs.update(embedding_kwargs)

    def compute_integration(self, adata):
        """
        Integrate data using SCANORAMA.

        Parameters
        ----------
        adata
            AnnData object. Results are stored in adata.obsm[self.embedding_key].
        """
        logging.info("Integrating data with scanorama")
        tmp = adata[adata.obs.sort_values(self.batch_key).index, :].copy()
        sc.external.pp.scanorama_integrate(
            tmp,
            key=self.batch_key,
            adjusted_basis=self.embedding_key,
            **self.method_kwargs,
        )
        adata.obsm[self.embedding_key] = tmp[adata.obs_names, :].obsm[self.embedding_key]

    def predict(self, adata):
        """
        Compute KNN classifier on SCANORAMA integrated data.

        Parameters
        ----------
        adata
            AnnData object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Saving knn on scanorama results to adata.obs["{self.result_key}"]')

        ref_idx = adata.obs["_labelled_train_indices"]
        train_X = adata[ref_idx].obsm[self.embedding_key]
        train_Y = adata.obs.loc[ref_idx, self.labels_key].cat.codes.to_numpy()

        knn = make_pipeline(
            PyNNDescentTransformer(n_neighbors=self.classifier_kwargs["n_neighbors"], n_jobs=settings.n_jobs),
            KNeighborsClassifier(metric="precomputed", weights=self.classifier_kwargs["weights"]),
        )

        knn.fit(train_X, train_Y)
        knn_pred = knn.predict(adata.obsm[self.embedding_key])

        # save_results
        adata.obs[self.result_key] = adata.uns["label_categories"][knn_pred]

        if self.return_probabilities:
            adata.obs[f"{self.result_key}_probabilities"] = np.max(
                knn.predict_proba(adata.obsm[self.embedding_key]), axis=1
            )

    def compute_umap(self, adata):
        """
        Compute UMAP embedding of integrated data.

        Parameters
        ----------
        adata
            AnnData object. Results are stored in adata.obsm[self.umap_key].
        """
        if self.compute_umap_embedding:
            logging.info(f'Saving UMAP of scanorama results to adata.obs["{self.embedding_key}"]')
            if settings.cuml:
                rsc.pp.neighbors(adata, use_rep=self.embedding_key)
                adata.obsm[self.umap_key] = rsc.tl.umap(adata, copy=True, **self.embedding_kwargs).obsm["X_umap"]
            else:
                sc.pp.neighbors(adata, use_rep=self.embedding_key)
                adata.obsm[self.umap_key] = sc.tl.umap(adata, copy=True, **self.embedding_kwargs).obsm["X_umap"]
