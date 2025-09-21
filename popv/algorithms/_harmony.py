from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from harmony import harmonize
from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from popv import settings

if settings.cuml:
    import rapids_singlecell as rsc
from popv.algorithms._base_algorithm import BaseAlgorithm


class KNN_HARMONY(BaseAlgorithm):
    """
    Class to compute KNN classifier after Harmony integration.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information. Default is "_batch_annotation".
    labels_key
        Key in obs field of adata for cell-type information. Default is "_labels_annotation".
    result_key
        Key in obs in which celltype annotation results are stored.
        Default is "popv_knn_harmony_prediction".
    embedding_key
        Key in obsm in which UMAP embedding of integrated data is stored.
        Default is "X_pca_harmony_popv".
    umap_key
        Key in obsm in which UMAP embedding of integrated data is stored.
        Default is "X_umap_harmony_popv".
    method_kwargs
        Additional parameters for HARMONY. Options at harmony.integrate_scanpy
    classifier_dict
        Dictionary to supply non-default values for KNN classifier. n_neighbors and weights supported.
    embedding_kwargs
        Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_knn_harmony_prediction",
        embedding_key: str | None = "X_pca_harmony_popv",
        umap_key: str | None = "X_umap_harmony_popv",
        method_kwargs: dict | None = None,
        classifier_dict: dict | None = None,
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
        if classifier_dict is None:
            classifier_dict = {}
        if method_kwargs is None:
            method_kwargs = {}

        self.method_kwargs = {"dimred": 50}
        if method_kwargs is not None:
            self.method_kwargs.update(method_kwargs)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

        self.embedding_kwargs = {"min_dist": 0.1}
        self.embedding_kwargs.update(embedding_kwargs)
        self.recompute_classifier = True

    def compute_integration(self, adata):
        logging.info("Integrating data with harmony")
        if (
            adata.uns["_prediction_mode"] == "inference"
            and self.embedding_key in adata.obsm
            and not settings.recompute_embeddings
        ):
            self.recompute_classifier = False
            index = joblib.load(os.path.join(adata.uns["_save_path_trained_models"], "pynndescent_index.joblib"))
            query_features = adata.obsm["X_pca"][adata.obs["_dataset"] == "query", :]
            indices, _ = index.query(query_features.astype(np.float32), k=5)
            neighbor_values = adata.obsm[self.embedding_key][adata.obs["_dataset"] == "ref", :][indices].astype(
                np.float32
            )
            adata.obsm[self.embedding_key][adata.obs["_dataset"] == "query", :] = np.mean(neighbor_values, axis=1)
            adata.obsm[self.embedding_key] = adata.obsm[self.embedding_key].astype(np.float32)
        elif adata.uns["_prediction_mode"] != "fast":
            adata.obsm[self.embedding_key] = harmonize(
                adata.obsm["X_pca"],
                adata.obs,
                batch_key=self.batch_key,
                use_gpu=settings.accelerator == "gpu",
            )
        else:
            raise ValueError(f"Prediction mode {adata.uns['_prediction_mode']} not supported for HARMONY")

    def predict(self, adata):
        """
        Predict celltypes using KNN on Harmony.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Saving knn on harmony results to adata.obs["{self.result_key}"]')

        if self.recompute_classifier:
            ref_idx = adata.obs["_labelled_train_indices"]

            train_X = adata[ref_idx].obsm[self.embedding_key].copy()
            train_Y = adata.obs.loc[ref_idx, self.labels_key].cat.codes.to_numpy()
            knn = make_pipeline(
                PyNNDescentTransformer(
                    n_neighbors=self.classifier_dict["n_neighbors"],
                    n_jobs=settings.n_jobs,
                ),
                KNeighborsClassifier(metric="precomputed", weights=self.classifier_dict["weights"]),
            )
            knn.fit(train_X, train_Y)
            if adata.uns["_prediction_mode"] == "retrain" and adata.uns["_save_path_trained_models"]:
                joblib.dump(
                    knn,
                    open(
                        os.path.join(
                            adata.uns["_save_path_trained_models"],
                            "harmony_knn_classifier.joblib",
                        ),
                        "wb",
                    ),
                )
        else:
            knn = joblib.load(
                open(
                    os.path.join(
                        adata.uns["_save_path_trained_models"],
                        "harmony_knn_classifier.joblib",
                    ),
                    "rb",
                )
            )

        # save_results
        embedding = adata[adata.obs["_predict_cells"] == "relabel"].obsm[self.embedding_key]
        knn_pred = knn.predict(embedding)
        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = adata.uns["label_categories"][
            knn_pred
        ]
        if self.return_probabilities:
            if f"{self.result_key}_probabilities" not in adata.obs.columns:
                adata.obs[f"{self.result_key}_probabilities"] = pd.Series(dtype="float64")
            adata.obs.loc[
                adata.obs["_predict_cells"] == "relabel",
                f"{self.result_key}_probabilities",
            ] = np.max(embedding, axis=1)

    def compute_umap(self, adata):
        """
        Compute UMAP embedding of integrated data.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obsm[self.umap_key].
        """
        if self.compute_umap_embedding:
            logging.info(f'Saving UMAP of harmony results to adata.obs["{self.umap_key}"]')
            if settings.cuml:
                rsc.pp.neighbors(adata, use_rep=self.embedding_key)
                adata.obsm[self.umap_key] = rsc.tl.umap(adata, copy=True, **self.embedding_kwargs).obsm["X_umap"]
            else:
                sc.pp.neighbors(adata, use_rep=self.embedding_key)
                adata.obsm[self.umap_key] = sc.tl.umap(adata, copy=True, **self.embedding_kwargs).obsm["X_umap"]
