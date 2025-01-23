from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import scipy
from OnClass.OnClassModel import OnClassModel

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class ONCLASS(BaseAlgorithm):
    """
    Class to compute OnClass cell-type prediction.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
        Default is "_batch_annotation".
    labels_key
        Key in obs field of adata for cell-type information.
        Default is "_labels_annotation".
    layer_key
        Layer in adata used for Onclass prediction.
        Default is adata.X.
    max_iter
        Maximum iteration in Onclass training.
        Default is 30.
    cell_ontology_obs_key
        Key in obs in which ontology celltypes are stored.
    result_key
        Key in obs in which celltype annotation results are stored.
        Default is "popv_onclass_prediction".
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layer_key: str | None = None,
        max_iter: int | None = 30,
        cell_ontology_obs_key: str | None = None,
        result_key: str | None = "popv_onclass_prediction",
        seen_result_key: str | None = "popv_onclass_seen",
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            seen_result_key=seen_result_key,
        )
        self.layer_key = layer_key
        self.cell_ontology_obs_key = cell_ontology_obs_key
        self.max_iter = max_iter
        self.labels_key = labels_key

    def predict(self, adata):
        """
        Predict celltypes using OnClass.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Computing Onclass. Storing prediction in adata.obs["{self.result_key}"]')
        adata.obs.loc[adata.obs["_dataset"] == "query", "self.labels_key"] = adata.uns["unknown_celltype_label"]

        train_idx = adata.obs["_ref_subsample"]

        if self.layer_key is None:
            train_x = adata[train_idx].X.copy()
        else:
            train_x = adata[train_idx].layers[self.layer_key].copy()
        if scipy.sparse.issparse(train_x):
            train_x = train_x.todense()

        cl_ontology_file = adata.uns["_cl_ontology_file"]
        nlp_emb_file = adata.uns["_nlp_emb_file"]
        train_model = OnClassModel(cell_type_nlp_emb_file=nlp_emb_file, cell_type_network_file=cl_ontology_file)

        if adata.uns["_save_path_trained_models"] is not None:
            model_path = os.path.join(adata.uns["_save_path_trained_models"], "OnClass")
        else:
            model_path = None

        if adata.uns["_prediction_mode"] == "retrain":
            train_y = adata[train_idx].obs[self.labels_key]
            _ = train_model.EmbedCellTypes(train_y)

            (
                corr_train_feature,
                corr_train_genes,
            ) = train_model.ProcessTrainFeature(
                train_x,
                train_y,
                adata.var_names,
                log_transform=False,
            )

            train_model.BuildModel(ngene=len(corr_train_genes))
            train_model.Train(
                corr_train_feature,
                train_y,
                save_model=model_path,
                max_iter=self.max_iter,
            )
        else:
            train_model.BuildModel(ngene=None, use_pretrain=model_path)

        subset = adata[adata.obs["_predict_cells"] == "relabel"]
        if self.layer_key is None:
            test_x = subset.X.copy()
        else:
            test_x = subset.layers[self.layer_key].copy()
        if self.return_probabilities:
            required_columns = {
                self.seen_result_key: pd.Series(index=subset.obs_names, dtype=str),
                self.result_key: pd.Series(index=subset.obs_names, dtype=str),
                f"{self.result_key}_probabilities": pd.Series(index=subset.obs_names, dtype=float),
                f"{self.seen_result_key}_probabilities": pd.Series(index=subset.obs_names, dtype=float),
            }
        else:
            required_columns = {
                self.seen_result_key: pd.Series(index=subset.obs_names, dtype=str),
                self.result_key: pd.Series(index=subset.obs_names, dtype=str),
            }

        result_df = pd.DataFrame(required_columns)
        shard_size = int(settings.shard_size)
        for i in range(0, subset.n_obs, shard_size):
            tmp_x = test_x[i : i + shard_size]
            if scipy.sparse.issparse(test_x):
                tmp_x = tmp_x.todense()
            names_x = subset.obs_names[i : i + shard_size]
            corr_test_feature = train_model.ProcessTestFeature(
                test_feature=tmp_x,
                test_genes=subset.var_names,
                use_pretrain=model_path,
                log_transform=False,
            )

            if adata.uns["_prediction_mode"] == "fast":
                onclass_pred = train_model.Predict(
                    corr_test_feature,
                    use_normalize=False,
                    refine=False,
                    unseen_ratio=-0.0,
                )
                onclass_seen = np.argmax(onclass_pred, axis=1)
                pred_label_str = [train_model.i2co[ind] for ind in onclass_seen]
                result_df.loc[names_x, self.result_key] = pred_label_str
                result_df.loc[names_x, self.seen_result_key] = pred_label_str
                if self.return_probabilities:
                    result_df.loc[names_x, f"{self.result_key}_probabilities"] = np.max(onclass_pred, axis=1)
                    result_df.loc[names_x, f"{self.seen_result_key}_probabilities"] = np.max(onclass_pred, axis=1)
            else:
                onclass_pred = train_model.Predict(
                    corr_test_feature,
                    use_normalize=False,
                    refine=True,
                    unseen_ratio=-1.0,
                )
                pred_label_str = [train_model.i2co[ind] for ind in onclass_pred[2]]
                result_df.loc[names_x, self.result_key] = pred_label_str

                onclass_seen = np.argmax(onclass_pred[0], axis=1)
                pred_label_str = [train_model.i2co[ind] for ind in onclass_seen]
                result_df.loc[names_x, self.seen_result_key] = pred_label_str

                if self.return_probabilities:
                    result_df.loc[names_x, f"{self.result_key}_probabilities"] = np.max(
                        onclass_pred[1], axis=1
                    ) / onclass_pred[1].sum(1)
                    result_df.loc[names_x, f"{self.seen_result_key}_probabilities"] = np.max(onclass_pred[0], axis=1)
        for col in required_columns.keys():
            if col not in adata.obs.columns:
                if "probabilities" in col:
                    adata.obs[col] = pd.Series(dtype="float64")  # Set dtype to float
                else:
                    adata.obs[col] = adata.uns["unknown_celltype_label"]
                    adata.obs[col] = adata.obs[col].astype(str)  # Set dtype to string
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", result_df.columns] = result_df
