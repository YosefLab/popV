from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class RF(BaseAlgorithm):
    """
    Class to compute Random forest classifier.

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
        result_key: str | None = "popv_rf_prediction",
        enable_cuml: bool = settings.cuml,
        classifier_dict: str | None = {},
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            layer_key=layer_key,
        )

        self.classifier_dict = {
            "class_weight": "balanced_subsample",
            "max_features": 200,
            "n_jobs": settings.n_jobs,
        }
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)
        self.enable_cuml = enable_cuml

    def _predict(self, adata):
        logging.info(
            f'Computing random forest classifier. Storing prediction in adata.obs["{self.result_key}"]'
        )

        test_x = adata[adata.obs["_predict_cells"] == "relabel"].layers[self.layer_key] if self.layer_key else adata.X

        if adata.uns["_prediction_mode"] == "retrain":
            # CUML RF doesn't support pickling, we always use sklearn RF
            train_idx = adata.obs["_ref_subsample"]
            train_x = (
                adata[train_idx].layers[self.layer_key]
                if self.layer_key
                else adata[train_idx].X
            )
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()
            rf = RandomForestClassifier(**self.classifier_dict)
            rf.fit(train_x, train_y)
            joblib.dump(
                rf,
                open(
                    os.path.join(adata.uns["_save_path_trained_models"], "rf_classifier.joblib"),
                    "wb",
                ),
            )
        else:
            rf = joblib.load(
                open(os.path.join(adata.uns["_save_path_trained_models"], "rf_classifier.joblib"), "rb")
            )

        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = adata.uns["label_categories"][
            rf.predict(test_x)
        ]
        if self.return_probabilities:
            if f"{self.result_key}_probabilities" not in adata.obs.columns:
                adata.obs[f"{self.result_key}_probabilities"] = pd.Series(dtype="float64")
            adata.obs.loc[adata.obs["_predict_cells"] == "relabel", f"{self.result_key}_probabilities"] = np.max(
                rf.predict_proba(test_x), axis=1
            ).astype(float)
