#!/usr/bin/env python
"""Random Forest model for TM-GSspinPlus spin multiplicities and oxidation states.

  * d_elec is recomputed with Metal.compute_valence_electron_count(m_ox), the
    value spin.py uses at prediction time; geom_nr from build_geometry_index
  * predicted multiplicities impossible for the electron count are replaced by
    the corresponding valid LS / IS / HS multiplicity (spin.correct_spin)
  * no Spin filter: TM-GSspinPlus labels (B3LYP*) are already pure LS/IS/HS

The total multiplicity model is shipped as TM-GSspinPlus_RandomForest.joblib.

Usage:
  python rf.py DATAFRAME METAL PROP MODE
    DATAFRAME    TM-GSspinPlus_rel_m_splitting.txt (tab separated)
    METAL        total | element symbol
    PROP         multiplicity | spin_multiplicity | m_ox
    MODE         free tag used in output file names
"""

import json
from sys import argv

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from cell2mol.classes.metal import Metal
from cell2mol.spin import build_geometry_index, predict_corrected, spin_type

SPIN_PROPS = ["spin_multiplicity", "multiplicity"]


def compute_valence_elec(label: str, m_ox: int) -> int:
    return Metal(label=label, coord=[0.0, 0.0, 0.0]).compute_valence_electron_count(
        m_ox
    )


def load_dataframe(path: str, metal: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter="\t")
    print("dataframe size:", len(df))
    print(df.columns)

    # ---- Metal filtering ----
    if metal != "total":
        if metal not in df["metal"].unique():
            print("No such metal in the database")
            exit()
        df = df[df["metal"] == metal]
    print("filtered size:", metal, len(df))

    # ---- Recompute features ----
    d_elec = [compute_valence_elec(m, int(ox)) for m, ox in zip(df.metal, df.m_ox)]
    if "d_elec" in df.columns:
        print("d_elec changed by recompute:", int((df.d_elec != d_elec).sum()))
    df["d_elec"] = d_elec
    df["geom_nr"] = df.geometry.map(build_geometry_index())
    df = df[df.geom_nr.notna()]
    if "hapticity" in df.columns:
        df["hapticity"] = df.hapticity.astype(int)
    return df


if __name__ == "__main__":
    dataframe, metal, prop, mode = argv[1:5]
    print("Sklearn version:", sklearn.__version__)

    df = load_dataframe(dataframe, metal)
    if "hapticity" in df.columns:
        print("Coordination complexes:", int((df.hapticity == 0).sum()))
        print("Complexes with haptic ligands:", int((df.hapticity == 1).sum()))

    # ---- Feature selection ----
    if prop in SPIN_PROPS:
        extract = ["elem_nr", "m_ox", "d_elec", "CN", "geom_nr", "rel_m"]
        bad = np.array(
            [
                spin_type(int(s), int(n)) == "invalid"
                for s, n in zip(df[prop], df.d_elec)
            ]
        )
        print("dropped impossible labels:", int(bad.sum()))
        df = df[~bad]
    elif prop == "m_ox":
        extract = ["elem_nr", "CN", "geom_nr", "rel_m"]
    else:
        print("No such property in the database")
        exit()
    if "hapticity" in df.columns:
        extract.append("hapticity")

    # ---- Build feature and label arrays ----
    Nfix = df.refcode.to_numpy(dtype=str)
    X = df[extract].to_numpy(dtype=np.float64)
    Y = df[prop].to_numpy(dtype=int)
    print("number of complexes:", len(Nfix))
    print("features:", extract)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # ---- Class statistics ----
    for uniq_prop in np.unique(Y):
        print(f"{prop} {uniq_prop}: {np.count_nonzero(Y == uniq_prop)}")

    # ---- Hyperparameter search ----
    random_grid = {
        "n_estimators": [100, 200, 300],
        "max_features": [0.25, 0.5, 0.75],
        "bootstrap": [True, False],
    }
    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(n_jobs=-1),
        param_distributions=random_grid,
        n_iter=18,
        cv=5,
        random_state=42,
    ).fit(X, Y)

    # ---- Cross-validation ----
    n_splits = 10
    acc_train = np.zeros(n_splits)
    acc_test = np.zeros(n_splits)
    acc_test_corrected = np.zeros(n_splits)
    f1_test_micro = np.zeros(n_splits)
    f1_test_macro = np.zeros(n_splits)
    n_corrected = 0
    oos = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for rep, (idx_tr, idx_te) in enumerate(skf.split(X, Y)):
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = Y[idx_tr], Y[idx_te]

        learner = rf_random.best_estimator_.fit(X_tr, y_tr)
        acc_train[rep] = 100 * accuracy_score(y_tr, learner.predict(X_tr))

        if prop in SPIN_PROPS:
            pred_te, corr_te = predict_corrected(learner, X_te)
        else:
            pred_te = corr_te = learner.predict(X_te)
        maxprob = np.max(learner.predict_proba(X_te), axis=1)

        acc_test[rep] = 100 * accuracy_score(y_te, pred_te)
        acc_test_corrected[rep] = 100 * accuracy_score(y_te, corr_te)
        f1_test_micro[rep] = f1_score(y_te, corr_te, average="micro")
        f1_test_macro[rep] = f1_score(y_te, corr_te, average="macro")
        n_corrected += int(np.count_nonzero(pred_te != corr_te))
        oos.extend(
            zip(
                Nfix[idx_te],
                X_te[:, 2].astype(int),
                y_te,
                pred_te,
                corr_te,
                maxprob.round(4),
            )
        )

    # ---- Summary ----
    print("\nSummary:")
    print(f"Train accuracy: {np.mean(acc_train):.3f} ± {np.std(acc_train):.3f}")
    print(f"Test accuracy (raw): {np.mean(acc_test):.3f} ± {np.std(acc_test):.3f}")
    print(
        f"Test accuracy (corrected): {np.mean(acc_test_corrected):.3f} ± {np.std(acc_test_corrected):.3f}"
    )
    print(f"Test f1 (micro, corrected): {np.mean(f1_test_micro):.3f}")
    print(f"Test f1 (macro, corrected): {np.mean(f1_test_macro):.3f}")
    print(f"Impossible predictions corrected: {n_corrected}")

    # ---- Save OOS predictions ----
    tag = f"{metal}_{prop}_{len(Y)}_{mode}"
    pd.DataFrame(
        oos, columns=["refcode", "d_elec", prop, "pred", "pred_corrected", "maxprob"]
    ).to_csv(f"{tag}_oos.csv", index=False)

    # ---- Train final model and save ----
    final_model = rf_random.best_estimator_.fit(X, Y)
    model_name = f"{tag}.joblib"
    dump(final_model, model_name)

    meta = {
        "sklearn_version": sklearn.__version__,
        "n_samples": len(Y),
        "features": extract,
        "classes": final_model.classes_.tolist(),
        "best_params": rf_random.best_params_,
    }
    with open(model_name.replace(".joblib", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Model saved:", model_name)
    print(
        "Feature importance:",
        dict(zip(extract, final_model.feature_importances_.round(4))),
    )
