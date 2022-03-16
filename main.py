from quickclus import QuickClus
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer


X, y = make_blobs(n_samples = 1000, n_features = 8, random_state = 1)

numerical_features = X[:, :6]
categorical_features = KBinsDiscretizer(n_bins = 3, encode = "ordinal").fit_transform(X[:, 6:])
categorical_features = np.where(
    categorical_features == 1.0,
    "M",
    np.where(categorical_features == 2.0, "H", "L"),
).astype(str)

numerical_columns = [f"num_{i}" for i in range(numerical_features.shape[1])]
df = pd.DataFrame(numerical_features, columns = numerical_columns)

clf = QuickClus(
    umap_combine_method = "intersection_union_mapper",
)

clf.fit(df)

print(clf.hdbscan_.labels_)

clf.tune_model()

results = clf.assing_results(df)

clf.cluster_summary(results)