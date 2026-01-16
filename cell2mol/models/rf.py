#!/usr/bin/env python

import sklearn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sys import argv
from joblib import dump
import json

from cell2mol.spin import make_geom_list

# ---- Input arguments ----
dataframe = argv[1]
metal = argv[2]
prop = argv[3]
mode = argv[4]

print("Sklearn version:", sklearn.__version__)

# ---- Load data ----
# df = pd.read_csv(dataframe)
df = pd.read_csv(dataframe, delimiter="\t")
print("dataframe size:", len(df))
print(df.columns)

df["geom_nr"] = ""
for tmc in df.refcode:
    df.loc[df.refcode == tmc, "geom_nr"] = make_geom_list()[
        df[df.refcode == tmc].geometry.item()
    ]

# ---- Metal filtering ----
if metal != "total":
    if metal in df["metal"].unique():
        df = df[df["metal"] == metal]
    else:
        print("No such metal in the database")
        exit()

print("filtered size:", metal, len(df))
if "hapticity" in df.columns:
    print("Coordination complexes:", len(df[df["hapticity"] == False]))
    print("Complexes with haptic ligands:", len(df[df["hapticity"] == True]))

# df = df[df["hapticity"] == False]
# print("filtered size:", metal, len(df))

# ---- Feature selection ----
if prop in ["spin_multiplicity", "multiplicity"]:
    extract = ["elem_nr", "m_ox", "d_elec", "CN", "geom_nr", "rel_m"]
    if "hapticity" in df.columns:
        extract.append("hapticity")
elif prop == "m_ox":
    extract = ["elem_nr", "CN", "geom_nr", "rel_m"]
    if "hapticity" in df.columns:
        extract.append("hapticity")
else:
    print("No such property in the database")
    exit()


# ---- Build feature and label arrays ----
Nfix = np.array(df["refcode"], dtype=str)
print("number of complexes:", len(Nfix))

X = np.vstack([np.array(df[df["refcode"] == name][extract]) for name in Nfix])
Y = np.array(
    [df[df.refcode == name][prop].item() for name in Nfix],
    dtype=int,
)

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

f1_train_micro = np.zeros(n_splits)
f1_test_micro = np.zeros(n_splits)
f1_train_macro = np.zeros(n_splits)
f1_test_macro = np.zeros(n_splits)
f1_train_weighted = np.zeros(n_splits)
f1_test_weighted = np.zeros(n_splits)

maxprob = []
l_oos = []

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for rep, (idx_tr, idx_te) in enumerate(skf.split(X, Y)):
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = Y[idx_tr], Y[idx_te]
    l_te = Nfix[idx_te]

    learner = rf_random.best_estimator_.fit(X_tr, y_tr)

    # Train metrics
    pred_tr = learner.predict(X_tr)
    acc_train[rep] = 100 * accuracy_score(y_tr, pred_tr)
    f1_train_micro[rep] = f1_score(y_tr, pred_tr, average="micro")
    f1_train_macro[rep] = f1_score(y_tr, pred_tr, average="macro")
    f1_train_weighted[rep] = f1_score(y_tr, pred_tr, average="weighted")

    # Test metrics
    pred_te = learner.predict(X_te)
    probs_te = np.around(learner.predict_proba(X_te), 4)

    acc_test[rep] = 100 * accuracy_score(y_te, pred_te)
    f1_test_micro[rep] = f1_score(y_te, pred_te, average="micro")
    f1_test_macro[rep] = f1_score(y_te, pred_te, average="macro")
    f1_test_weighted[rep] = f1_score(y_te, pred_te, average="weighted")

    maxprob.extend(np.max(probs_te, axis=1))
    l_oos.extend(l_te)


# ---- Summary ----
print("\nSummary:")
print(f"Train accuracy: {np.mean(acc_train):.3f} ± {np.std(acc_train):.3f}")
print(f"Test accuracy: {np.mean(acc_test):.3f} ± {np.std(acc_test):.3f}")
print(f"Train f1 (micro): {np.mean(f1_train_micro):.3f}")
print(f"Test f1 (micro): {np.mean(f1_test_micro):.3f}")


# ---- Save OOS probabilities ----
dat = np.column_stack((np.array(l_oos, dtype=object), np.array(maxprob)))
np.savetxt(f"{metal}_{prop}_maxprob_{len(df)}_{mode}.txt", dat, fmt="%s")


# ---- Train final model and save ----
final_model = rf_random.best_estimator_.fit(X, Y)

model_name = f"{metal}_{prop}_{len(df)}.joblib"
dump(final_model, model_name)

meta = {
    "sklearn_version": sklearn.__version__,
    "n_samples": len(df),
    "features": extract,
}

with open(model_name.replace(".joblib", "_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Model saved:", model_name)
print("Feature importance:", final_model.feature_importances_)
