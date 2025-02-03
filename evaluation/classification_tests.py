import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

from sklearn.metrics import PrecisionRecallDisplay

from sklearn.decomposition import PCA


EMBEDDINGS = [
    ["../data/seq_feats.csv", None],
    ["../data/yeast_emb_only_embeddings.csv", None],
    ["../data/yeast_emb_embeddings_yeastnet.csv", None],
    ["../data/yeast_emb_embeddings_genex.csv", None],
    ["../data/yeast_emb_embeddings_yeastnet_genex.csv", None],
]
CLASSIFIERS = [
    sklearn.ensemble.RandomForestClassifier(100), 
    sklearn.linear_model.LogisticRegression(max_iter=500)
]

classes = pd.read_csv("../data/Costanzo_classes.csv").set_index("gene_id")
print(f"Non-essential genes {(classes == 0).sum().iloc[0]/len(classes)*100:.2f}%")

dataset_names = []
datasets = []
for emb_path, proc in EMBEDDINGS:
    dataset_names.append(emb_path.rstrip(".csv").split("/")[-1])
    emb = pd.read_csv(emb_path, index_col=0).set_index("gene_id")

    if proc is not None:
        emb = pd.DataFrame(proc.fit_transform(emb), index=emb.index)

    dataset = emb.merge(classes, left_index=True, right_index=True)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    datasets.append([X, y])

fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, (X, y)) in enumerate(zip(dataset_names, datasets)):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    for classifier in CLASSIFIERS:
        # fit deletes previous attributes
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(f"{name} {classifier}")
        print("Acc\tPrec\tRec\tF1")
        print(f"{sklearn.metrics.accuracy_score(y_test, y_pred):.4f}\t{sklearn.metrics.precision_score(y_test, y_pred):.4f}\t{sklearn.metrics.recall_score(y_test, y_pred):.4f}\t{sklearn.metrics.f1_score(y_test, y_pred):.4f}")

        y_proba = classifier.predict_proba(X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name=f"{classifier} on {name}")

ax.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()