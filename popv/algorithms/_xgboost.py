from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class XGboost(BaseAlgorithm):
    """
    Class to compute Xgboost classifier.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
    labels_key
        Key in obs field of adata for cell-type information.
    layer_key
        Key in layers field of adata used for classification. By default uses 'X' (log1p10K).
    result_key
        Key in obs in which celltype annotation results are stored.
    enable_cuml
        Enable cuml, which currently doesn't support weighting. Default to popv.settings.cuml.
    classifier_dict
        Dictionary to supply non-default values for RF classifier. Options at sklearn.ensemble.RandomForestClassifier.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layer_key: str | None = None,
        result_key: str | None = "popv_xgboost_prediction",
        classifier_dict: str | None = {},
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            layer_key=layer_key,
        )

        self.classifier_dict = {
            "tree_method": "hist",
            "device": "cuda" if settings.cuml else "cpu",
            "objective": "multi:softprob",
        }
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

    def _predict(self, adata):
        logging.info(
            f'Computing XGboost classifier. Storing prediction in adata.obs["{self.result_key}"]'
        )

        subset = adata[adata.obs["_predict_cells"] == "relabel"]
        test_x = subset.layers[self.layer_key] if self.layer_key else subset.X
        test_y = subset.obs[self.labels_key].cat.codes.to_numpy()
        dtest = xgb.DMatrix(test_x, test_y)

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = (
                adata[train_idx].layers[self.layer_key]
                if self.layer_key
                else adata[train_idx].X
            )
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()
            dtrain = xgb.DMatrix(train_x, train_y)
            self.classifier_dict["num_class"] = len(adata.uns["label_categories"])

            bst = xgb.train(self.classifier_dict, dtrain, num_boost_round=300)
            bst.save_model(os.path.join(adata.uns["_save_path_trained_models"], "xgboost_classifier.model"))
        else:
            bst = xgb.Booster({"device": "cuda" if False else "cpu"})
            bst.load_model(os.path.join(adata.uns["_save_path_trained_models"], "xgboost_classifier.model"))

        output_probabilities = bst.predict(dtest)
        unassigned_idx = list(adata.uns["label_categories"]).index("unassigned")
        output_probabilities[:, unassigned_idx] = 0.
        if self.result_key not in adata.obs.columns:
             adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = adata.uns["label_categories"][
            np.argmax(output_probabilities, axis=1)
        ]
        if self.return_probabilities:
            if f"{self.result_key}_probabilities" not in adata.obs.columns:
                adata.obs[f"{self.result_key}_probabilities"] = pd.Series(dtype="float64")
            adata.obs.loc[adata.obs["_predict_cells"] == "relabel", f"{self.result_key}_probabilities"] = np.max(
                output_probabilities, axis=1
            ).astype(float)
