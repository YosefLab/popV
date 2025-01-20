from __future__ import annotations

import logging
import os

import faiss
import numpy as np
import scanpy as sc
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class BBKNN(BaseAlgorithm):
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_knn_on_bbknn_prediction",
        embedding_key: str | None = "X_bbknn_umap_popv",
        method_kwargs: dict | None = None,
        classifier_kwargs: dict | None = None,
        embedding_kwargs: dict | None = None,
    ) -> None:
        """
        Class to compute KNN classifier after BBKNN integration.

        Parameters
        ----------
        batch_key
            Key in obs field of adata for batch information.
        labels_key
            Key in obs field of adata for cell-type information.
        result_key
            Key in obs in which celltype annotation results are stored.
        embedding_key
            Key in obsm in which UMAP embedding of integrated data is stored.
        method_kwargs
            Additional parameters for BBKNN. Options at sc.external.pp.bbknn
        classifier_kwargs
            Dictionary to supply non-default values for KNN classifier. Options at sklearn.neighbors.KNeighborsClassifier
        embedding_kwargs
            Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
        """
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
        )
        if embedding_kwargs is None:
            embedding_kwargs = {}
        if classifier_kwargs is None:
            classifier_kwargs = {}
        if method_kwargs is None:
            method_kwargs = {}

        self.method_kwargs = {
            "metric": "euclidean",
            "approx": False,
            "n_pcs": 50,
            "neighbors_within_batch": 3,
            "use_annoy": False,
        }
        if method_kwargs is not None:
            self.method_kwargs.update(method_kwargs)

        self.classifier_kwargs = {"weights": "uniform", "n_neighbors": 15}
        if classifier_kwargs is not None:
            self.classifier_kwargs.update(classifier_kwargs)

        self.embedding_kwargs = {"min_dist": 0.1}
        self.embedding_kwargs.update(embedding_kwargs)

    def _compute_integration(self, adata):
        logging.info("Integrating data with bbknn")
        if False:  # adata.uns["_prediction_mode"] == "inference" and "X_umap_bbknn" in adata.obsm and not settings.recompute_embeddings:
            index = faiss.read_index(os.path.join(adata.uns["_save_path_trained_models"], "faiss_index.faiss"))
            query_features = adata.obsm["X_pca"][adata.obs["_dataset"] == "query", :]
            _, indices = index.search(query_features.astype(np.float32), 5)

            neighbor_embedding = adata.obsm["X_umap_bbknn"][adata.obs["_dataset"] == "ref", :][indices].astype(
                np.float32
            )
            adata.obsm["X_umap_bbknn"][adata.obs["_dataset"] == "query", :] = np.mean(neighbor_embedding, axis=1)
            adata.obsm["X_umap_bbknn"] = adata.obsm["X_umap_bbknn"].astype(np.float32)

            neighbor_probabilities = adata.obs[f"{self.result_key}_probabilities"][adata.obs["_dataset"] == "ref", :][
                indices
            ].astype(np.float32)
            adata.obs.loc[adata.obs["_dataset"] == "query", f"{self.result_key}_probabilities"] = np.mean(
                neighbor_probabilities, axis=1
            )

            neighbor_prediction = adata.obs[f"{self.result_key}"][adata.obs["_dataset"] == "ref", :][indices].astype(
                np.float32
            )
            adata.obs.loc[adata.obs["_dataset"] == "query", f"{self.result_key}"] = mode(neighbor_prediction, axis=1)
        else:
            if len(adata.obs[self.batch_key].unique()) > 100:
                logging.warning("Using PyNNDescent instead of FAISS as high number of batches leads to OOM.")
                sc.external.pp.bbknn(adata, batch_key=self.batch_key, use_faiss=False, use_rep="X_pca")
            else:
                sc.external.pp.bbknn(adata, batch_key=self.batch_key, use_faiss=True, use_rep="X_pca")

    def _predict(self, adata):
        logging.info(f'Saving knn on bbknn results to adata.obs["{self.result_key}"]')

        distances = adata.obsp["distances"]
        ref_idx = adata.obs["_labelled_train_indices"]
        ref_dist_idx = np.where(ref_idx)[0]
        train_y = adata.obs.loc[ref_idx, self.labels_key].cat.codes.to_numpy()
        train_distances = distances[ref_dist_idx, :][:, ref_dist_idx]
        test_distances = distances[:, :][:, ref_dist_idx]

        # Make sure BBKNN found the required number of neighbors, otherwise reduce n_neighbors for KNN.
        smallest_neighbor_graph = np.min(
            [
                np.diff(test_distances.indptr).min(),
                np.diff(train_distances.indptr).min(),
            ]
        )
        if smallest_neighbor_graph < 15:
            logging.warning(f"BBKNN found only {smallest_neighbor_graph} neighbors. Reduced neighbors in KNN.")
            self.classifier_kwargs["n_neighbors"] = smallest_neighbor_graph

        knn = KNeighborsClassifier(metric="precomputed", **self.classifier_kwargs)
        knn.fit(train_distances, y=train_y)
        adata.obs[self.result_key] = adata.uns["label_categories"][knn.predict(test_distances)]

        if self.return_probabilities:
            adata.obs[f"{self.result_key}_probabilities"] = np.max(knn.predict_proba(test_distances), axis=1)

    def _compute_embedding(self, adata):
        if self.compute_embedding:
            logging.info(f'Saving UMAP of bbknn results to adata.obs["{self.embedding_key}"]')
            if len(adata.obs[self.batch_key]) < 30 and settings.cuml:
                method = "rapids"
            else:
                logging.warning("Using UMAP instead of RAPIDS as high number of batches leads to OOM.")
                method = "umap"
                # RAPIDS not possible here as number of batches drastically increases GPU RAM.
            adata.obsm[self.embedding_key] = sc.tl.umap(adata, copy=True, method=method, **self.embedding_kwargs).obsm[
                "X_umap"
            ]
