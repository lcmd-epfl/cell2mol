#!/usr/bin/env python

import os
import os.path
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib
import pickle
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
import ast
from cell2mol.spin import make_geom_list


dataframe=argv[1]
metal = argv[2]
mode = argv[3]

print("Sklearn version:", sklearn.__version__)

df = pd.read_csv(dataframe, delimiter=",")
print("the length of dataframe", len(df))

print(df.columns)

df["geom_nr"] = ""
for tmc in df.refcode :
    df.loc[df.refcode==tmc, "geom_nr"] = make_geom_list()[df[df.refcode==tmc].geometry.item()]


if metal == "total":
    pass
else :
    if metal in df["metal"].unique():
        df = df[df["metal"] == metal]
    else :
        print("No such metal in the database")
        exit()

print("the length of", metal,  len(df))
#print(df[:10])

#exit()

# prop = "spin_multiplicity"
prop = "m_ox"
# extract = ["elem_nr", "m_ox", "d_elec"] # F_TM
# extract = ["CN", "geom_nr", "rel_m"] # F_CE
# extract = ["elem_nr", "m_ox", "d_elec", "CN", "geom_nr", "rel_m"] # F_TM+CE
extract = ["elem_nr", "d_elec", "CN", "geom_nr", "rel_m"] # for metal oxidation state prediction m_ox_v1
# extract = ["elem_nr", "CN", "geom_nr", "rel_m"] # for metal oxidation state prediction m_ox_v2

Nfix=list(df["refcode"])
print("the number of complexes :", len(Nfix))
Nfix=np.array(Nfix, dtype=str)

X = np.vstack([np.array(df[df["refcode"] == name][extract]) for name in Nfix])
print(extract)
print("feature size", X.shape)
Y = np.array( [ df[df.refcode==name][prop].item() for name in Nfix] , dtype=int,)
print("reference data", Y.shape)


# spinlist = np.unique(Y)
# for spin in spinlist:
#     nc = np.count_nonzero(Y == spin)
#     print("In database, the number of spin {} entries is {}".format(spin, nc))

m_ox_list = np.unique(Y)
for m_ox in m_ox_list:
    nc = np.count_nonzero(Y == m_ox)
    print("In database, the number of m_ox {} entries is {}".format(m_ox, nc))

# Set up printing options
run_diagnosis = True
print_incorrect = True
print_uncertain = True
extra_analysis = False

n_estimators = [100, 200, 300]
max_features = [0.25, 0.5, 0.75]
bootstrap = [True, False]
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "bootstrap": bootstrap,
}
rf_random = RandomizedSearchCV(
    estimator=RandomForestClassifier(n_jobs=-1),
    param_distributions=random_grid,
    n_iter=18,
    verbose=0,
    cv=5,
    random_state=42,
).fit(X, Y)


n_splits = 10
acc_train = np.zeros((n_splits))
acc_test = np.zeros((n_splits))

f1_train_micro = np.zeros((n_splits)) 
f1_test_micro = np.zeros((n_splits))
f1_train_macro = np.zeros((n_splits))
f1_test_macro = np.zeros((n_splits))
f1_train_weighted = np.zeros((n_splits))
f1_test_weighted = np.zeros((n_splits))

maxprob = []
l_oos = []
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
for rep, (idx_tr, idx_te) in enumerate(skf.split(X, Y)):
    X_tr = np.take(X, idx_tr, axis=0)
    X_te = np.take(X, idx_te, axis=0)
    y_tr = np.take(Y, idx_tr)
    y_te = np.take(Y, idx_te)
    l_tr = np.take(Nfix, idx_tr)
    l_te = np.take(Nfix, idx_te)

    # Initializing the learner
    learner = rf_random.best_estimator_.fit(X_tr, y_tr)
    predictions = learner.predict(X_tr)
    prediction_probs = np.around(learner.predict_proba(X_tr), 4)
    is_correct = predictions == y_tr
    acc_train[rep] = 100 * accuracy_score(y_tr, predictions)
    f1_train_micro[rep] = f1_score(y_tr, predictions, average="micro")
    f1_train_macro[rep] = f1_score(y_tr, predictions, average="macro")
    f1_train_weighted[rep] = f1_score(y_tr, predictions, average="weighted")
    
    # Test set predict
    predictions = learner.predict(X_te)
    prediction_probs = np.around(learner.predict_proba(X_te), 4)
    is_correct = predictions == y_te
    acc_test[rep] = 100 * accuracy_score(y_te, predictions)
    f1_test_micro[rep] = f1_score(y_te, predictions, average="micro")
    f1_test_macro[rep] = f1_score(y_te, predictions, average="macro")
    f1_test_weighted[rep] = f1_score(y_te, predictions, average="weighted")


    is_certain = [True if np.max(probs) >= 0.5 else False for probs in prediction_probs]
    l_oos.extend(l_te)
    #print(prediction_probs)
    #print(np.amax(prediction_probs, axis=1))
    maxprob.extend(list(np.amax(prediction_probs, axis=1)))
    if print_incorrect:
        print(f"\n Incorrect predictions for replica {rep}:")
    for idx, sys in enumerate(is_correct):
        if not sys and print_incorrect:
            print(
                f"System {l_te[idx]} has prediction {predictions[idx]} with probability {np.max(prediction_probs[idx])} and reference {y_te[idx]} metal {df[df.refcode==l_te[idx]]['metal'].item()}"
            )

print("\n \n Summary of replica results:")
print(f"Training mean accuracy was {round(np.mean(acc_train),3)} with STD {round(np.std(acc_train),3)}")
print(f"Test mean accuracy was {round(np.mean(acc_test),3)} with STD {round(np.std(acc_test),3)}")
print(f"Training mean f1_score_micro was {np.mean(f1_train_micro)} with STD {np.std(f1_train_micro)}")
print(f"Test mean f1_score_micro was {np.mean(f1_test_micro)} with STD {np.std(f1_test_micro)}")
print(f"Training mean f1_score_macro was {np.mean(f1_train_macro)} with STD {np.std(f1_train_macro)}")
print(f"Test mean f1_score_macro was {np.mean(f1_test_macro)} with STD {np.std(f1_test_macro)}")
print(f"Training mean f1_score_weighted was {np.mean(f1_train_weighted)} with STD {np.std(f1_train_weighted)}")
print(f"Test mean f1_score_weighted was {np.mean(f1_test_weighted)} with STD {np.std(f1_test_weighted)}")

try:
    assert len(maxprob) == len(l_oos)
except AssertionError:
    print(len(maxprob), len(l_oos))
maxprob = np.array(maxprob)
l_oos = np.array(l_oos, dtype=object)
dat = np.column_stack((l_oos, maxprob))
np.savetxt(metal + "_maxprob_{}.txt".format(mode), dat, delimiter=" ", fmt="%s")

# We train a model on all available data and save it
filename = "{}_{}.pkl".format(metal, mode)
learner = learner = rf_random.best_estimator_.fit(X, Y)
pickle.dump(learner, open(filename, "wb"))
print("feature importance", learner.feature_importances_)

exit()

# Not doing out of sample things because there is no out of sample
if run_diagnosis:

    # Initializing the learner
    learner = learner = rf_random.best_estimator_.fit(X, Y)

    predictions = learner.predict(X)
    prediction_probs = np.around(learner.predict_proba(X), 2)
    is_correct = predictions == Y
    is_certain = [True if np.max(probs) >= 0.5 else False for probs in prediction_probs]
    n_correct = np.count_nonzero(is_correct)
    print(
        "\nWith final model, agreement in training is: %f %%"
        % (100 * float(n_correct) / Y.shape[0])
    )
    if print_incorrect:
        print("\n Incorrect predictions:")
    for idx, sys in enumerate(is_correct):
        if not sys and print_incorrect:
            print(
                f"System {Nfix[idx]} has prediction {predictions[idx]} with probability {np.max(prediction_probs[idx])} and reference {Y[idx]}"
            )
    if print_uncertain:
        print("\n Uncertain predictions:")
    for idx, sys in enumerate(is_certain):
        if not sys and print_uncertain:
            print(
                f"System {Nfix[idx]} has prediction {predictions[idx]} with probability {np.max(prediction_probs[idx])} and reference {Y[idx]}"
            )
    if extra_analysis:
        from treeinterpreter import treeinterpreter as ti

        prediction, bias, contributions = ti.predict(learner, X)
        for idx, sys in enumerate(Nfix):
            print(f"System {Nfix[idx]} has contributions {contributions[0,idx]}:")

# trained_model = pickle.load(open(filename, 'rb'))

exit()
