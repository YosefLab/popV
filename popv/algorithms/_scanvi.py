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
    Class to compute classifier in scANVI model.

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
        Default is "popv_scanvi_prediction".
    embedding_key
        Key in obsm in which latent embedding is stored.
        Default is "X_scanvi_popv".
    umap_key
        Key in obsm in which UMAP embedding of integrated data is stored.
        Default is "X_umap_scanvi_popv".
    model_kwargs
        Dictionary to supply non-default values for SCVI model. Options at :class:`scvi.model.SCANVI`.
    classifier_kwargs
        Dictionary to supply non-default values for SCANVI classifier.
        Options at classifier_paramerers in :class:`scvi.model.SCANVI`.
    embedding_kwargs
        Dictionary to supply non-default values for UMAP embedding. Options at :func:`scanpy.tl.umap`.
    train_kwargs
        Dictionary to supply non-default values for training scvi. Options at :meth:`scvi.model.SCANVI.train`.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        save_folder: str | None = None,
        result_key: str | None = "popv_scanvi_prediction",
        embedding_key: str | None = "X_scanvi_popv",
        umap_key: str | None = "X_umap_scanvi_popv",
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
        self.umap_key = umap_key
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

    def compute_integration(self, adata):
        """
        Compute scANVI model and integrate data.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obsm[self.embedding_key].
        """
        logging.info("Integrating data with scANVI")
        if adata.uns["_prediction_mode"] == "retrain":
            if adata.uns["_pretrained_scvi_path"]:
                scvi_model = scvi.model.SCVI.load(
                    os.path.join(adata.uns["_save_path_trained_models"], "scvi"),
                    adata=adata,
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
                    devices=[settings.device] if settings.cuml else settings.n_jobs,
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
        latent_representation = self.model.get_latent_representation()
        relabel_indices = adata.obs["_predict_cells"] == "relabel"
        if self.embedding_key not in adata.obsm:
            # Initialize X_scanvi with the correct shape if it doesn't exist
            adata.obsm[self.embedding_key] = np.zeros((adata.n_obs, latent_representation.shape[1]))
        adata.obsm[self.embedding_key][relabel_indices, :] = latent_representation

    def predict(self, adata):
        """
        Predict celltypes using scANVI.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Saving scanvi label prediction to adata.obs["{self.result_key}"]')

        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = self.model.predict(
            adata[adata.obs["_predict_cells"] == "relabel"]
        )
        if self.return_probabilities:
            if f"{self.result_key}_probabilities" not in adata.obs.columns:
                adata.obs[f"{self.result_key}_probabilities"] = pd.Series(dtype="float64")
            if f"{self.result_key}_probabilities" not in adata.obsm:
                adata.obsm[f"{self.result_key}_probabilities"] = pd.DataFrame(
                    np.nan,
                    index=adata.obs_names,
                    columns=adata.uns["label_categories"][:-1],
                )
            probs = self.model.predict(adata[adata.obs["_predict_cells"] == "relabel"], soft=True)
            adata.obs.loc[adata.obs["_predict_cells"] == "relabel", f"{self.result_key}_probabilities"] = np.max(
                probs, axis=1
            )
            adata.obsm[f"{self.result_key}_probabilities"].loc[adata.obs["_predict_cells"] == "relabel", :] = probs

    def compute_umap(self, adata):
        """
        Compute UMAP embedding of integrated data.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obsm[self.umap_key].
        """
        if self.compute_umap_embedding:
            logging.info(f'Saving UMAP of BBKNN results to adata.obsm["{self.umap_key}"]')
            if settings.cuml:
                import rapids_singlecell as rsc

                rsc.pp.neighbors(adata, use_rep=self.embedding_key)
                adata.obsm[self.umap_key] = rsc.tl.umap(adata, copy=True, **self.embedding_kwargs).obsm["X_umap"]
            else:
                sc.pp.neighbors(adata, use_rep=self.embedding_key)
                adata.obsm[self.umap_key] = sc.tl.umap(adata, copy=True, **self.embedding_kwargs).obsm["X_umap"]
