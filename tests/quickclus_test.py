#Libraries--------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer
import random

import unittest

from quickclus.QuickClus import QuickClus

#Adding some random nulls--------------------------------------------------------------------
def add_random_nulls(df, null_prop = 0.1):
    ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    for row, col in random.sample(ix, int(round(null_prop * len(ix)))):
        df.iat[row, col] = np.nan

    return df


#Unit test------------------------------------------------------------------------------------

class QuickClusTestCase(unittest.TestCase):

    def setUp(self):
        #Numerical + categorical
        n_clusters = 3

        #Include random null values
        null_prop = 0.1
        add_nulls = True

        X, y = make_blobs(n_samples = 1000, n_features = 8, random_state = 1)

        numerical_features = X[:, :6]
        categorical_features = KBinsDiscretizer(n_bins = 3, encode = "ordinal").fit_transform(X[:, 6:])
        categorical_features = np.where(
            categorical_features == 1.0,
            "M",
            np.where(categorical_features == 2.0, "H", "L"),
        ).astype(str)

        numerical_columns = [f"num_{i}" for i in range(numerical_features.shape[1])]
        self.df = pd.DataFrame(numerical_features, columns = numerical_columns)

        categorical_columns = [f"cat_{i}" for i in range(categorical_features.shape[1])]
        for idx, c in enumerate(categorical_columns):
            self.df[c] = categorical_features[:, idx]

        self.df_numerical = pd.DataFrame(numerical_features, columns = numerical_columns)
        self.df_categorical = pd.DataFrame(categorical_features, columns = categorical_columns)


        if add_nulls:
            self.df = add_random_nulls(df = self.df, null_prop = null_prop)
            self.df_numerical = add_random_nulls(df = self.df_numerical, null_prop = null_prop)
            self.df_categorical = add_random_nulls(df = self.df_categorical, null_prop = null_prop)

        #Numerical + categorical
        self.clf = QuickClus(
            n_components = 2,
            random_state = 42,
            n_neighbors = 10,
            umap_combine_method = "intersection_union_mapper",
        )


    #Tests
    def test_fit_categorical(self):
        self.clf.fit(self.df)
        self.assertEqual(self.clf.umap_categorical_.embedding_.shape, (len(self.df), self.clf.n_components))

    def test_fit_numerical(self):
        self.clf.fit(self.df)
        self.assertEqual(self.clf.umap_numerical_.embedding_.shape, (len(self.df), self.clf.n_components))

    def test_fit_only_categorical(self):
        self.clf.fit(self.df_categorical)
        self.assertEqual(self.clf.umap_categorical_.embedding_.shape, (len(self.df_categorical), self.clf.n_components))

    def test_fit_only_numerical(self):
        self.clf.fit(self.df_numerical)
        self.assertEqual(self.clf.umap_numerical_.embedding_.shape, (len(self.df_numerical), self.clf.n_components))

    def test_umap_embeddings(self):
        self.clf.fit(self.df)
        self.assertEqual(self.clf.umap_combined.embedding_.shape, (len(self.df), self.clf.n_components))

    def test_hdbscan_labels(self):
        self.clf.fit(self.df)
        self.assertEqual(self.clf.hdbscan_.labels_.shape[0], self.df.shape[0])

    def test_hdbscan_optimization(self):
        self.clf.fit(self.df)
        self.clf.tune_model(n_trials = 5, min_cluster_start = 0.01, min_cluster_end = 0.05,
                    min_samples_start = 0.01, min_samples_end = 0.05, max_epsilon = None)
        self.assertEqual(self.clf.hdbscan_.labels_.shape[0], self.df.shape[0])

    def test_quickclus_fit_is_df(self):
        with pytest.raises(TypeError):
            self.clf.fit([1, 2, 3])

    def test_quickclus_method(self):
        with pytest.raises(KeyError):
            _ = QuickClus(umap_combine_method = "notamethod").fit(self.df)

    def test_repr(self):
        self.assertEqual(str(type(self.clf.__repr__)), "<class 'method'>")

    def test_condensed_tree(self):
        self.clf.fit(self.df)
        self.clf.plot_condensed_tree()

    def test_embedding_labels(self):
        self.clf.fit(self.df)
        self.clf.plot_embedding_labels()

    def test_2d_labels(self):
        self.clf.fit(self.df)
        self.clf.plot_2d_labels()

    def test_assing_results(self):
        self.clf.fit(self.df)
        self.results = self.clf.assing_results(self.df)
        self.assertEqual(self.df.shape[0], self.results.shape[0])
        self.assertEqual(self.df.shape[1] + 1, self.results.shape[1])



if __name__ == '__main__':
    unittest.main()







