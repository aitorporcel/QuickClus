## QuickClus

QuickClus is a Python module for clustering categorical and numerical data using [UMAP](https://github.com/lmcinnes/umap) and [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan). 
QuickClus allows incorporating numerical and categorical values (even with null values) into the clustering, in a simple and fast way. The imputation of null values, the scaling and transformation of numerical variables, and the combination of categorical variables are performed automatically.

## Installation

```bash
python3 -m pip install QuickClus
```

## Usage

QuickClus requires a Pandas dataframe as input, which may contain numeric, categorical, or both types of variables. In the case of null values, QuickClus takes care of the imputation and subsequent scaling of all the features. All this process is done automatically under the hood.
It is also possible to automatically optimize the algorithm using optuna, calling tune_model().
Finally, QuickClus provides a summary of the characteristics of each cluster.

```python
from quickclus import QuickClus
clf = QuickClus(
    umap_combine_method = "intersection_union_mapper",
)

clf.fit(df)

print(clf.hdbscan_.labels_)

clf.tune_model()

results = clf.assing_results(df)

clf.cluster_summary(results)

```

## Examples

TO DO

## References

```bibtex
@article{mcinnes2018umap-software,
  title={UMAP: Uniform Manifold Approximation and Projection},
  author={McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas},
  journal={The Journal of Open Source Software},
  volume={3},
  number={29},
  pages={861},
  year={2018}
}
```

```bibtex
@article{mcinnes2017hdbscan,
  title={hdbscan: Hierarchical density based clustering},
  author={McInnes, Leland and Healy, John and Astels, Steve},
  journal={The Journal of Open Source Software},
  volume={2},
  number={11},
  pages={205},
  year={2017}
}
```