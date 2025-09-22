from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scvi.model import SCVI

from popv import settings
from popv._faiss_knn_classifier import FAISSKNNProba
from popv.algorithms._base_algorithm import BaseAlgorithm


class KNN_SCVI(BaseAlgorithm):
    """
    Class to compute KNN classifier after scVI integration.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
        Default is "_batch_annotation".
    labels_key
        Key in obs field of adata for cell-type information.
        Default is "_labels_annotation".
    max_epochs
        Number of epochs scvi is trained.
    result_key
        Key in obs in which celltype annotation results are stored.
        Default is "popv_knn_scvi_prediction".
    embedding_key
        Key in obsm in which latent dimensions are stored.
        Default is "X_scvi_popv".
    umap_key
        Key in obsm in which UMAP embedding of integrated data is stored.
        Default is "X_umap_scvi_popv".
    model_kwargs
        Dictionary to supply non-default values for SCVI model.
        Options at :class:`scvi.model.SCVI`.
        Default is
        {"n_layers": 3, "n_latent": 20, "gene_likelihood": "nb", "use_batch_norm": "none", "use_layer_norm": "both", "encode_covariates": True}.
    classifier_kwargs
        Dictionary to supply non-default values for KNN classifier.
        See :class:`sklearn.neighbors.KNeighborsClassifier`.
        Default is {"weights": "uniform", "n_neighbors": 15}.
    embedding_kwargs
        Dictionary to supply non-default values for UMAP embedding.
        See :func:`scanpy.tl.umap`. Default is {"min_dist": 0.1}.
    train_kwargs
        Dictionary to supply non-default values for training scVI.
        Options at :meth:`scvi.model.SCVI.train`.
        Default is
        {"max_epochs": 20, "batch_size": 512, "accelerator": settings.accelerator, "plan_kwargs": {"n_epochs_kl_warmup": 20}}.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        save_folder: str | None = None,
        result_key: str | None = "popv_knn_on_scvi_prediction",
        embedding_key: str | None = "X_scvi_popv",
        umap_key: str | None = "X_umap_scvi_popv",
        model_kwargs: dict | None = None,
        classifier_dict: dict | None = None,
        embedding_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
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
        if model_kwargs is None:
            model_kwargs = {}
        if train_kwargs is None:
            train_kwargs = {}
        self.save_folder = save_folder

        self.model_kwargs = {
            "n_layers": 3,
            "n_latent": 20,
            "gene_likelihood": "nb",
            "use_batch_norm": "none",
            "use_layer_norm": "both",
            "encode_covariates": True,
        }

        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

        self.train_kwargs = {
            "max_epochs": 20,
            "batch_size": 512,
            "accelerator": settings.accelerator,
            "plan_kwargs": {"n_epochs_kl_warmup": 20},
        }
        self.train_kwargs.update(train_kwargs)
        self.max_epochs = train_kwargs.get("max_epochs", None)

        self.embedding_kwargs = {"min_dist": 0.3}
        if embedding_kwargs is not None:
            self.embedding_kwargs.update(embedding_kwargs)

    def compute_integration(self, adata):
        """
        Compute scVI integration.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obsm[self.embedding_key].
        """
        logging.info("Integrating data with scvi")
        if not adata.uns["_pretrained_scvi_path"]:
            SCVI.setup_anndata(
                adata,
                batch_key=self.batch_key,
                labels_key=self.labels_key,
                layer="scvi_counts",
            )
            model = SCVI(adata, **self.model_kwargs)
            logging.info("Training scvi offline.")
        else:
            query = adata[adata.obs["_predict_cells"] == "relabel"].copy()
            model = SCVI.load_query_data(query, adata.uns["_pretrained_scvi_path"])
            logging.info("Training scvi online.")

        if adata.uns["_prediction_mode"] == "fast":
            self.train_kwargs["max_epochs"] = 1
            model.train(**self.train_kwargs)
        else:
            if self.max_epochs is None:
                self.max_epochs = min(round((20000 / adata.n_obs) * 200), 200)
            model.train(**self.train_kwargs)

            if adata.uns["_save_path_trained_models"] and adata.uns["_prediction_mode"] == "retrain":
                save_path = os.path.join(adata.uns["_save_path_trained_models"], "scvi")
                # Update scvi for scanvi.
                adata.uns["_pretrained_scvi_path"] = save_path
                model.save(
                    save_path,
                    save_anndata=False,
                    overwrite=True,
                )

        latent_representation = model.get_latent_representation()
        relabel_indices = adata.obs["_predict_cells"] == "relabel"
        if self.embedding_key not in adata.obsm:
            # Initialize X_scanvi with the correct shape if it doesn't exist
            adata.obsm[self.embedding_key] = np.zeros((adata.n_obs, latent_representation.shape[1]))
        adata.obsm[self.embedding_key][relabel_indices, :] = latent_representation

    def predict(self, adata):
        """
        Predict celltypes using KNN on scVI embedding.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Saving knn on scvi results to adata.obs["{self.result_key}"]')
        knn = FAISSKNNProba(n_neighbors=self.classifier_dict["n_neighbors"])

        if adata.uns["_prediction_mode"] == "retrain":
            ref_idx = adata.obs["_labelled_train_indices"]

            train_X = adata[ref_idx].obsm[self.embedding_key].copy()
            train_Y = adata.obs.loc[ref_idx, self.labels_key].cat.codes.to_numpy()

            knn.fit(train_X, train_Y)
            knn.save(
                os.path.join(adata.uns["_save_path_trained_models"], "scvi_knn_classifier"),
            )
        else:
            knn = knn.load(
                os.path.join(adata.uns["_save_path_trained_models"], "scvi_knn_classifier"),
            )

        # save_results
        embedding = adata[adata.obs["_predict_cells"] == "relabel"].obsm[self.embedding_key]
        knn_pred = knn.predict(embedding, adata.uns["label_categories"][:-1])
        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = knn_pred
        if self.return_probabilities:
            if f"{self.result_key}_probabilities" not in adata.obs.columns:
                adata.obs[f"{self.result_key}_probabilities"] = pd.Series(dtype="float64")
            if f"{self.result_key}_probabilities" not in adata.obsm:
                adata.obsm[f"{self.result_key}_probabilities"] = pd.DataFrame(
                    np.nan,
                    index=adata.obs_names,
                    columns=adata.uns["label_categories"][:-1],
                )
            probabilities = knn.predict_proba(embedding, adata.uns["label_categories"][:-1])
            adata.obs.loc[
                adata.obs["_predict_cells"] == "relabel",
                f"{self.result_key}_probabilities",
            ] = np.max(probabilities, axis=1)
            adata.obsm[f"{self.result_key}_probabilities"].loc[adata.obs["_predict_cells"] == "relabel", :] = (
                probabilities
            )

    def compute_umap(self, adata):
        """
        Compute UMAP embedding of scVI results.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obsm[self.umap_key].
        """
        if self.compute_umap_embedding:
            logging.info(f'Saving UMAP of scvi results to adata.obs["{self.umap_key}"]')
            if settings.cuml:
                import rapids_singlecell as rsc

                rsc.pp.neighbors(adata, use_rep=self.embedding_key)
                adata.obsm[self.umap_key] = rsc.tl.umap(adata, copy=True, **self.embedding_kwargs).obsm["X_umap"]
            else:
                sc.pp.neighbors(adata, use_rep=self.embedding_key)
                adata.obsm[self.umap_key] = sc.tl.umap(adata, copy=True, **self.embedding_kwargs).obsm["X_umap"]
