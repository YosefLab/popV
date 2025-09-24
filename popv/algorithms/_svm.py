from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class Support_Vector(BaseAlgorithm):
    """
    Class to compute LinearSVC.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
        Default is "_batch_annotation".
    labels_key
        Key in obs field of adata for cell-type information.
        Default is "_labels_annotation".
    layer_key
        Key in layers field of adata used for classification. By default uses 'X' (log1p10K).
    result_key
        Key in obs in which celltype annotation results are stored.
        Default is "popv_svm_prediction".
    classifier_dict
        Dictionary to supply non-default values for SVM classifier. Options at
        :class:`sklearn.svm.LinearSVC`.
        Default is {'C': 1, 'max_iter': 5000, 'class_weight': 'balanced'}.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layer_key: str | None = None,
        result_key: str | None = "popv_svm_prediction",
        classifier_dict: str | None = {},
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
        )

        self.layer_key = layer_key
        self.classifier_dict = {
            "C": 1,
            "max_iter": 5000,
            "class_weight": "balanced",
        }
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

    def predict(self, adata):
        """
        Predict celltypes using LinearSVC.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Computing support vector machine. Storing prediction in adata.obs["{self.result_key}"]')
        subset = adata[adata.obs["_predict_cells"] == "relabel", :]
        test_x = subset.layers[self.layer_key] if self.layer_key else subset.X

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = adata[train_idx].layers[self.layer_key] if self.layer_key else adata[train_idx].X
            train_x = np.array(train_x.todense())
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()
            if settings.cuml:
                from cuml.svm import LinearSVC as CuMLLinearSVC

                self.classifier_dict["probability"] = self.return_probabilities
                cuml_clf = CuMLLinearSVC(**self.classifier_dict)
                cuml_clf.fit(train_x, train_y)
                self.classifier_dict.pop("probability")
                sk_clf = svm.LinearSVC()
                sk_clf.coef_ = cuml_clf.coef_.to_output("numpy")
                sk_clf.intercept_ = cuml_clf.intercept_.to_output("numpy")
                sk_clf.classes_ = cuml_clf.classes_.to_output("numpy")
                clf = CalibratedClassifierCV(sk_clf, cv="prefit")
            else:
                clf = CalibratedClassifierCV(svm.LinearSVC(**self.classifier_dict))
            clf.fit(train_x, train_y)
            joblib.dump(
                clf,
                open(
                    os.path.join(
                        adata.uns["_save_path_trained_models"],
                        "svm_classifier.joblib",
                    ),
                    "wb",
                ),
            )

        if self.return_probabilities:
            required_columns = [self.result_key, f"{self.result_key}_probabilities"]
            results_df_probabilities = pd.DataFrame(
                index=subset.obs_names, columns=adata.uns["label_categories"][:-1], dtype=float
            )
        else:
            required_columns = [self.result_key]

        result_df = pd.DataFrame(index=subset.obs_names, columns=required_columns, dtype=float)
        result_df[self.result_key] = result_df[self.result_key].astype("object")
        clf = joblib.load(
            open(
                os.path.join(adata.uns["_save_path_trained_models"], "svm_classifier.joblib"),
                "rb",
            )
        )
        result_df[self.result_key] = adata.uns["label_categories"][clf.predict(test_x).astype(int)]
        if self.return_probabilities:
            probabilities = clf.predict_proba(test_x)
            result_df[f"{self.result_key}_probabilities"] = np.max(probabilities, axis=1)
            results_df_probabilities = probabilities

        for col in required_columns:
            if col not in adata.obs.columns:
                adata.obs[col] = (
                    pd.Series(dtype="float64") if "probabilities" in col else adata.uns["unknown_celltype_label"]
                )
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", result_df.columns] = result_df
        if self.return_probabilities:
            if f"{self.result_key}_probabilities" not in adata.obsm.keys():
                adata.obsm[f"{self.result_key}_probabilities"] = pd.DataFrame(
                    index=adata.obs_names, columns=adata.uns["label_categories"][:-1], dtype=float
                )
            adata.obsm[f"{self.result_key}_probabilities"].loc[adata.obs["_predict_cells"] == "relabel", :] = (
                results_df_probabilities
            )
