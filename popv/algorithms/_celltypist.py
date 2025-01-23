from __future__ import annotations

import logging
import os

import celltypist
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mode

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class CELLTYPIST(BaseAlgorithm):
    """
    Class to compute Celltypist classifier.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
        Default is "_batch_annotation".
    labels_key
        Key in obs field of adata for cell-type information.
        Default is "_labels_annotation".
    result_key
        Key in obs in which celltype annotation results are stored.
        Default is "popv_celltypist_prediction".
    method_kwargs
        Additional parameters for celltypist training.
        Options at :func:`celltypist.train`.
    classifier_dict
        Dictionary to supply non-default values for celltypist annotation.
        Options at :func:`celltypist.annotate`.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_celltypist_prediction",
        method_kwargs: dict | None = None,
        classifier_dict: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
        )

        if classifier_dict is None:
            classifier_dict = {}
        if method_kwargs is None:
            method_kwargs = {}

        self.method_kwargs = {"check_expression": False, "n_jobs": 10, "max_iter": 500}
        if method_kwargs is not None:
            self.method_kwargs.update(method_kwargs)

        self.classifier_dict = {"mode": "best match", "majority_voting": True}
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

    def predict(self, adata):
        """
        Predict celltypes using Celltypist.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Saving celltypist results to adata.obs["{self.result_key}"]')

        if adata.uns["_prediction_mode"] == "fast":
            self.classifier_dict["majority_voting"] = False
            over_clustering = None
        elif (
            adata.uns["_prediction_mode"] == "inference"
            and "over_clustering" in adata.obs
            and not settings.recompute_embeddings
        ):
            index = joblib.load(os.path.join(adata.uns["_save_path_trained_models"], "pynndescent_index.joblib"))
            query_features = adata.obsm["X_pca"][adata.obs["_dataset"] == "query", :]
            indices, _ = index.query(query_features.astype(np.float32), k=5)
            neighbor_values = adata.obs.loc[adata.obs["_dataset"] == "ref", "over_clustering"].cat.codes.values[indices]
            adata.obs.loc[adata.obs["_dataset"] == "query", "over_clustering"] = adata.obs[
                "over_clustering"
            ].cat.categories[mode(neighbor_values, axis=1).mode.flatten()]
            over_clustering = adata.obs.loc[adata.obs["_predict_cells"] == "relabel", "over_clustering"]
        else:
            flavor = "rapids" if settings.cuml else "vtraag"
            transformer = "rapids" if settings.cuml else None
            sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", transformer=transformer)
            sc.tl.louvain(adata, resolution=25.0, key_added="over_clustering", flavor=flavor)
            over_clustering = adata.obs.loc[adata.obs["_predict_cells"] == "relabel", "over_clustering"]

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            if len(train_idx) > 100000 and not settings.cuml:
                self.method_kwargs["use_SGD"] = True
                self.method_kwargs["mini_batch"] = True

            train_adata = adata[train_idx].copy()
            model = celltypist.train(
                train_adata,
                self.labels_key,
                use_GPU=settings.cuml,
                **self.method_kwargs,
            )

            model.write(os.path.join(adata.uns["_save_path_trained_models"], "celltypist.pkl"))
        predictions = celltypist.annotate(
            adata[adata.obs["_predict_cells"] == "relabel"],
            model=os.path.join(adata.uns["_save_path_trained_models"], "celltypist.pkl"),
            over_clustering=over_clustering,
            **self.classifier_dict,
        )
        out_column = (
            "majority_voting" if "majority_voting" in predictions.predicted_labels.columns else "predicted_labels"
        )

        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = predictions.predicted_labels[
            out_column
        ]
        if self.return_probabilities:
            if f"{self.result_key}_probabilities" not in adata.obs.columns:
                adata.obs[f"{self.result_key}_probabilities"] = pd.Series(dtype="float64")
            adata.obs.loc[
                adata.obs["_predict_cells"] == "relabel",
                f"{self.result_key}_probabilities",
            ] = predictions.probability_matrix.max(axis=1).values
