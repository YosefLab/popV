from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class SCANVI_POPV(BaseAlgorithm):
    """
    Class to compute classifier in scANVI model and predict labels.

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
    model_kwargs
        Dictionary to supply non-default values for SCVI model. Options at scvi.model.SCVI
    classifier_kwargs
        Dictionary to supply non-default values for SCANVI classifier.
        Options at classifier_paramerers in scvi.model.SCANVI.from_scvi_model.
    embedding_kwargs
        Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
    train_kwargs
        Dictionary to supply non-default values for training scvi. Options at scvi.model.SCVI.train
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        save_folder: str | None = None,
        result_key: str | None = "popv_scanvi_prediction",
        embedding_key: str | None = "X_scanvi_umap_popv",
        model_kwargs: dict | None = None,
        classifier_kwargs: dict | None = None,
        embedding_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
        )
        self.save_folder = save_folder

        if embedding_kwargs is None:
            embedding_kwargs = {}
        if classifier_kwargs is None:
            classifier_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if train_kwargs is None:
            train_kwargs = {}

        self.model_kwargs = {
            "dropout_rate": 0.05,
            "dispersion": "gene",
            "n_layers": 3,
            "n_latent": 20,
            "gene_likelihood": "nb",
            "use_batch_norm": "none",
            "use_layer_norm": "both",
            "encode_covariates": True,
        }
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)

        self.train_kwargs = {
            "max_epochs": 20,
            "batch_size": 512,
            "n_samples_per_label": 20,
            "accelerator": settings.accelerator,
            "plan_kwargs": {"n_epochs_kl_warmup": 20},
            "max_epochs_unsupervised": 20,
        }
        self.train_kwargs.update(train_kwargs)
        self.max_epochs_unsupervised = self.train_kwargs.pop("max_epochs_unsupervised")
        self.max_epochs = self.train_kwargs.get("max_epochs", None)

        self.classifier_kwargs = {"n_layers": 3, "dropout_rate": 0.1}
        if classifier_kwargs is not None:
            self.classifier_kwargs.update(classifier_kwargs)

        self.embedding_kwargs = {"min_dist": 0.3}
        self.embedding_kwargs.update(embedding_kwargs)

    def _compute_integration(self, adata):
        logging.info("Integrating data with scANVI")
        if adata.uns["_prediction_mode"] == "retrain":
            if adata.uns["_pretrained_scvi_path"]:
                scvi_model = scvi.model.SCVI.load(
                    os.path.join(adata.uns["_save_path_trained_models"], "scvi"), adata=adata
                )
            else:
                scvi.model.SCVI.setup_anndata(
                    adata,
                    batch_key=self.batch_key,
                    labels_key=self.labels_key,
                    layer="scvi_counts",
                )
                scvi_model = scvi.model.SCVI(adata, **self.model_kwargs)
                scvi_model.train(
                    max_epochs=self.max_epochs_unsupervised,
                    accelerator=settings.accelerator,
                    plan_kwargs={"n_epochs_kl_warmup": 20},
                )

            self.model = scvi.model.SCANVI.from_scvi_model(
                scvi_model,
                unlabeled_category=adata.uns["unknown_celltype_label"],
                classifier_parameters=self.classifier_kwargs,
            )
        else:
            query = adata[adata.obs["_predict_cells"] == "relabel"].copy()
            self.model = scvi.model.SCANVI.load_query_data(
                query,
                os.path.join(adata.uns["_save_path_trained_models"], "scanvi"),
                freeze_classifier=True,
            )

        if adata.uns["_prediction_mode"] == "fast":
            self.train_kwargs.update({"max_epochs": 1})
        self.model.train(**self.train_kwargs)
        if adata.uns["_prediction_mode"] == "retrain":
            self.model.save(
                os.path.join(adata.uns["_save_path_trained_models"], "scanvi"),
                save_anndata=False,
                overwrite=True,
            )

    def _predict(self, adata):
        logging.info(
            f'Saving scanvi label prediction to adata.obs["{self.result_key}"]'
        )

        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = self.model.predict(
            adata[adata.obs["_predict_cells"] == "relabel"])
        if self.return_probabilities:
            if f"{self.result_key}_probabilities" not in adata.obs.columns:
                adata.obs[f"{self.result_key}_probabilities"] = pd.Series(dtype="float64")
            adata.obs.loc[adata.obs["_predict_cells"] == "relabel", f"{self.result_key}_probabilities"] = np.max(
                self.model.predict(adata[adata.obs["_predict_cells"] == "relabel"], soft=True), axis=1
            )

    def _compute_embedding(self, adata):
        if self.compute_embedding:
            logging.info(
                f'Saving UMAP of scanvi results to adata.obs["{self.embedding_key}"]'
            )
            # Update the .obsm["X_scanvi"] only for the relevant rows
            latent_representation = self.model.get_latent_representation()
            relabel_indices = adata.obs["_predict_cells"] == "relabel"
            if "X_scanvi" not in adata.obsm:
                # Initialize X_scanvi with the correct shape if it doesn't exist
                adata.obsm["X_scanvi"] = np.zeros((adata.n_obs, latent_representation.shape[1]))
            adata.obsm["X_scanvi"][relabel_indices, :] = latent_representation
            transformer = "rapids" if settings.cuml else None
            sc.pp.neighbors(adata, use_rep="X_scanvi", transformer=transformer)
            method = "rapids" if settings.cuml else "umap"
            adata.obsm[self.embedding_key] = sc.tl.umap(
                adata, copy=True, method=method, **self.embedding_kwargs
            ).obsm["X_umap"]
