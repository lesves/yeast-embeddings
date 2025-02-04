import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

from sklearn.metrics import PrecisionRecallDisplay

#from sklearn.decomposition import PCA


EMBEDDINGS = [
    ["../data/seq_feats.csv", "Seq. feats"],
    ["../data/yeast_emb_only_embeddings.csv", "Base"],
    ["../data/yeast_emb_embeddings_yeastnet.csv", "Base + YeastNet"],
    ["../data/yeast_emb_embeddings_genex.csv", "Base + Gene ex."],
    ["../data/yeast_emb_embeddings_yeastnet_genex.csv", "Base + YeastNet + Gene ex."],
    ["../data/vae.csv", "VAE"],
    ["../data/cvae.csv", "CVAE"],
]
CLASSIFIERS = [
    sklearn.linear_model.LogisticRegression(max_iter=1000, n_jobs=8),
    sklearn.ensemble.RandomForestClassifier(100),
]

classes = pd.read_csv("../data/Costanzo_classes.csv").set_index("gene_id")
essential_ratio = (classes == 0).sum().iloc[0]/len(classes)
print(f"Non-essential genes {essential_ratio*100:.2f}%")

dataset_names = []
datasets = []
for emb_path, emb_name in EMBEDDINGS:
    dataset_names.append(emb_name)
    emb = pd.read_csv(emb_path, index_col=0)
    if "gene_id" in emb:
        emb = emb.set_index("gene_id")

    dataset = emb.merge(classes, left_index=True, right_index=True)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    datasets.append([X, y])

fig, ax = plt.subplots(1, len(CLASSIFIERS), figsize=(14, 5), constrained_layout=True)


#cm = plt.get_cmap("gist_rainbow")
#num_colors = len(EMBEDDINGS)
#styles = ["dashed", "solid"]
#assert len(styles) == len(CLASSIFIERS)
#ax.set_prop_cycle(color=[cm(1.*i/num_colors) for i in range(num_colors) for j in range(len(styles))])

for i, (name, (X, y)) in enumerate(zip(dataset_names, datasets)):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    for j, classifier in enumerate(CLASSIFIERS):
        ax[j].set_title(f"{classifier.__class__.__name__}")
        # fit deletes previous attributes
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(f"{name} {classifier}")
        print("Acc\tPrec\tRec\tF1")
        print(f"{sklearn.metrics.accuracy_score(y_test, y_pred):.4f}\t{sklearn.metrics.precision_score(y_test, y_pred):.4f}\t{sklearn.metrics.recall_score(y_test, y_pred):.4f}\t{sklearn.metrics.f1_score(y_test, y_pred):.4f}")

        y_proba = classifier.predict_proba(X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[j], name=name)

for j in range(len(CLASSIFIERS)):
    PrecisionRecallDisplay.from_predictions(y_test, np.zeros_like(y_test), ax=ax[j], name=f"Chance level", linestyle="dotted", color="grey")
    ax[j].legend(bbox_to_anchor=(1, 1), fontsize="x-small")

plt.tight_layout()

plt.savefig("fig.png", bbox_inches="tight")
plt.savefig("fig.eps", bbox_inches="tight", format="eps")
plt.show()
