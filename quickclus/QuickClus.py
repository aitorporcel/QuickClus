#Libraries-----------------------------------------------------------------------------------
import logging
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer, StandardScaler, QuantileTransformer

import umap

import hdbscan
import optuna

from sklearn.base import BaseEstimator, ClassifierMixin

from quickclus.utils import *

#Logs
logger = logging.getLogger("quickclus")
logger.setLevel(logging.ERROR)
sh = logging.StreamHandler()
sh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger.addHandler(sh)


class QuickClus(BaseEstimator, ClassifierMixin):
    """QuickClus
    
    Creates UMAP embeddings and HDSCAN clusters from a pandas DataFrame with mixed data

    Parameters
    ----------
        random_state : int, default = None
            Random State for both UMAP and numpy.random.
            If set to None UMAP will run in Numba in multicore mode but
            results may vary between runs.
            Setting a seed may help to offset the stochastic nature of
            UMAP by setting it with fixed random seed.

        n_neighbors: int, default = 15
            Level of neighbors for UMAP.
            Setting this higher will generate higher densities at the expense
            of requiring more computational complexity.

        min_cluster_size: int, default = 15
            Minimum Cluster size for HDBSCAN.
            The minimum number of points from which a cluster needs to be
            formed.
        
        min_samples : int, default = None
            Samples used for HDBSCAN.
            The larger this is set the more noise points get declared and the
            more restricted clusters become to only dense areas.
            If None, min_samples = min_cluster_size

        threshold_combine_rare_levels: float, default = 0.02
            To avoid an excessive increase in dimensionality when transforming
            categorical variables into-one hot encoding, rare levels can be combined.
            This value indicates the minimum proportion of a category
            that should not be combined into "other".

        n_components: int, default = None
            Number of components for UMAP.
            These are dimensions to reduce the data down to.
            Ideally, this needs to be a value that preserves all the information
            to form meaningful clusters. Default is the logarithm of total
            number of features.

        imputer_strategy_numerical: str, default = "mean"
            Imputation strategy for numerical variables.
            The values can be: "mean", "median", "most_frequent"

        scaler_type_numerical: str, default = "standard"
            Scaler strategy for numerical variables.
            The values can be: "robust" (RobustScaler), "standard" (StandardScaler)

        transformation_type_numerical: str, default = "power"
            Scaler strategy for numerical variables.
            The values can be: "power" (PowerTransformer), "quantile" (QuantileTransformer)

        umap_combine_method: str, default = "intersection"
            Method by which to combine embeddings spaces.
            Options include: intersection, union, contrast,
            intersection_union_mapper
            The latter combines both the intersection and union of
            the embeddings.
            See: https://umap-learn.readthedocs.io/en/latest/composing_models.html

        n_neighbors_intersection_union: int, default = None
            Level of neighbors for UMAP to use to combine umaps embeddings
            if umap_combine_method = "intersection_union_mapper"
            If None, n_neighbors_intersection_union = n_neighbors

        verbose: bool, defualt = False
            Level of verbosity to print when fitting and predicting.
            Setting to False will only show Warnings that appear.

    """
    def __init__(self,
                random_state: int = None,
                n_neighbors: int = 15,
                min_cluster_size: int = 15,
                min_samples: int = None,
                threshold_combine_rare_levels: float = 0.02,
                n_components: int = None,
                scaler_type_numerical: str = "standard",
                imputer_strategy_numerical: str = "mean",
                transformation_type_numerical: str = "power",
                umap_combine_method: str = "intersection",
                n_neighbors_intersection_union: int = None,
                verbose: bool = False, ):

        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.threshold_combine_rare_levels = threshold_combine_rare_levels
        self.n_components = n_components
        self.scaler_type_numerical = scaler_type_numerical
        self.imputer_strategy_numerical = imputer_strategy_numerical
        self.transformation_type_numerical = transformation_type_numerical
        self.umap_combine_method = umap_combine_method
        self.n_neighbors_intersection_union = n_neighbors_intersection_union

        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
            logger.setLevel(logging.ERROR)
            self.verbose = False
            # supress deprecation warnings
            # see: https://stackoverflow.com/questions/54379418
            
            def noop(*args, **kargs):
                    pass

            warnings.warn = noop

        if isinstance(random_state, int):
            np.random.seed(seed = random_state)
        else:
            logger.info("No random seed passed, running UMAP in Numba")

        if min_samples is None:
            self.min_samples = min_cluster_size
            logger.info("No min_samples passed, using min_samples = min_cluster_size")

        if (n_neighbors_intersection_union is None) & (umap_combine_method == "intersection_union_mapper"):
            self.n_neighbors_intersection_union = n_neighbors
            logger.info("No n_neighbors_intersection_union passed, using n_neighbors_intersection_union = n_neighbors")



    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit function for call UMAP and HDBSCAN

        Parameters
        ----------
            df : pandas DataFrame
                DataFrame object with named columns of categorical and numerics

        Returns
        -------
            Fitted: None
                Fitted UMAPs and HDBSCAN
        """

        check_is_df(df)

        if not isinstance(self.n_components, int):
            self.n_components = int(round(np.log(df.shape[1])))

        logger.info("Extracting categorical features")
        self.categorical_ = self._extract_categorical_data(df)

        #If the dataset has categorical columns:
        if self.categorical_.shape[1] > 0:
            logger.info("Preprocessing categorical features")
            self._preprocess_categorical_data()

            logger.info("Transforming categorical features into UMAP")
            self._transform_categorical_umap()
        else:
            logger.info("No categorical features in the dataset")


        
        logger.info("Extracting numerical features")
        self.numerical_ = self._extract_numerical_data(df)

        #If the dataset has categorical columns:
        if self.numerical_.shape[1] > 0:
            logger.info("Preprocessing categorical features")
            self._preprocess_numerical_data()

            logger.info("Transforming numerical features into UMAP")
            self._transform_numerical_umap()
        else:
            logger.info("No numerical features in the dataset")

        #Combine the data
        logger.info("Mapping/Combining Embeddings")
        if (self.numerical_.shape[1] > 0) & (self.categorical_.shape[1] > 0):
            self._combine_umap_data()
        elif (self.numerical_.shape[1] > 0):
            self.umap_combined = self.umap_numerical_
        elif self.categorical_.shape[1] > 0:
            self.umap_combined = self.umap_categorical_
        else:
            raise TypeError("No numerical or categorical data were found")


        logger.info("Fitting HDBSCAN...")
        self._fit_hdbscan()



    def _extract_categorical_data(self, data):
        """
        Extracts the categorical data from the dataframe
        
        Parameters
        ----------
            data : pandas DataFrame
                DataFrame object with named columns of categorical and numerics

        Returns
        -------
            categorical_data: pandas DataFrame
                pandas DataFrame with the categorical variables
        
        """

        #Select only the categorical columns
        categorical_data = data.select_dtypes(exclude = ["float", "int", "datetime"])

        return categorical_data

    def _preprocess_categorical_data(self):
        """
        Preprocess the categorical data: Rare level combination, na imputation with the mode and one hot encoding
        
        Parameters
        ----------
            self.categorical_ : pandas DataFrame
                pandas DataFrame with categorical features
            self.threshold_combine_rare_levels : float
                Minimum proportion of a category to not be combined


        Returns
        -------
            self.preprocessed_categorical_: numpy.array
                numpy array with the preprocessed categorical data
        
        """

        #Combine rare levels into "other"
        for category in self.categorical_.columns:
            self.categorical_[category] = self.categorical_[category].mask(self.categorical_[category].map(self.categorical_[category].value_counts(normalize = True)) <= self.threshold_combine_rare_levels, 'Other')


        #Use a simple imputer with the mode and one hot encoding
        imputer_cat = SimpleImputer(strategy = "most_frequent")
        one_hot = OneHotEncoder(categories = "auto", handle_unknown = "ignore")

        #Create the pipeline and transform the data
        categorical_pipeline = Pipeline([("imputer", imputer_cat),
                                ("one_hot", one_hot)])

        preprocessed_cat = categorical_pipeline.fit_transform(self.categorical_)

        self.preprocessed_categorical_ = preprocessed_cat

        return self


    def _transform_categorical_umap(self):
        """
        Transforms the preprocessed categorical data into a umap embedding
        
        Parameters
        ----------
            self.preprocessed_categorical_ : scipy.sparse.csr.csr_matrix
                matrix with preprocessed categorical data

            self.n_neighbors : int
                number of neighbors UMAP

            self.n_components: int
                number of components UMAP

            self.random_state: int
                seed

        Returns
        -------
            self.umap_categorical_: umap.umap_.UMAP
                categorical umap embedding

        """
        #TODO: In some cases dice doesn't work. Check why.

        logger.info(f"Preprocessed categorical data shape: {self.preprocessed_categorical_.shape}")

        try:
            categorical_umap = umap.UMAP(
                    metric = "dice",
                    n_neighbors = self.n_neighbors,
                    n_components = self.n_components,
                    min_dist = 0.0,
                    random_state = self.random_state,
                ).fit(self.preprocessed_categorical_)

            logger.info("Metric used for categorical data: dice")

        except:
            categorical_umap = umap.UMAP(
                    metric = "jaccard",
                    n_neighbors = self.n_neighbors,
                    n_components = self.n_components,
                    min_dist = 0.0,
                    random_state = self.random_state,
                ).fit(self.preprocessed_categorical_)
            logger.info("Metric used for categorical data: jaccard")

        self.umap_categorical_ = categorical_umap

        return self


    def _extract_numerical_data(self, data):
        """
        Extracts the numerical data from the dataframe
        
        Parameters
        ----------
            data : pandas DataFrame
                DataFrame object with named columns of categorical and numerics

        Returns
        -------
            numerical_data: pandas DataFrame
                pandas DataFrame with the numerical variables

        """

        numerical_data = data.select_dtypes(include = ["float", "int"])

        return numerical_data


    def _preprocess_numerical_data(self):
        """
        Preprocess of numerical data: na imputation, scaler, and transformation
        
        Parameters
        ----------
            self.numerical_ : pandas DataFrame
                pandas DataFrame with numerical features

            self.imputer_strategy_numerical: str
                imputation strategy, 'mean', 'median', 'most_frequent'
            
            self.scaler_type_numerical: str
                scaler type, 'standard' or 'robust'

            self.transformation_type_numerical: str
                transformation type, 'power' or 'quantile'

        Returns
        -------
            self.preprocessed_numerical_: numpy.array
                numpy array with the preprocessed numerical data
        """

        #Imputer
        imputer_numeric = SimpleImputer(strategy = self.imputer_strategy_numerical)

        #Scaler
        if self.scaler_type_numerical == "robust":
            scaler_numeric = RobustScaler()
        elif self.scaler_type_numerical == "standard":
            scaler_numeric = StandardScaler()
        else:
            raise Exception("Select a valid scaler")

        #Transformation
        if self.transformation_type_numerical == "power":
            transform_numeric = PowerTransformer()
        elif self.transformation_type_numerical == "quantile":
            transform_numeric = QuantileTransformer()
        else:
            raise Exception("Select a valid transformation type")


        #Pipeline
        numerical_pipeline = Pipeline([("imputer", imputer_numeric),
                                    ("scaler", scaler_numeric),
                                    ("transform", transform_numeric)])

        
        self.preprocessed_numerical_  = numerical_pipeline.fit_transform(self.numerical_)

        return self


    def _transform_numerical_umap(self):
        """
        Transforms the preprocessed numerical data into a umap embedding

        Parameters
        ----------
            self.preprocessed_numerical_: scipy.sparse.csr.csr_matrix
                matrix with preprocessed numerical data

            self.n_neighbors: int
                number of neighbors UMA

            self.n_components: int
                number of components UMAP

            self.random_state: int
                seed

        Returns
        -------
            self.umap_numerical_: umap.umap_.UMAP
                umap embedding
        """
        
        
        numerical_umap = umap.UMAP(
            metric = "l2",
            n_neighbors = self.n_neighbors,
            n_components = self.n_components,
            min_dist = 0.0,
            random_state = self.random_state,
        ).fit(self.preprocessed_numerical_)

        self.umap_numerical_ = numerical_umap

        return self



    def _combine_umap_data(self):
        """
        Combines the numerical and categorical data embeddings
        
        Parameters
        ----------
            self.umap_numerical_ : umap.umap_.UMAP
                numerical data embedding

            self.umap_categorical_: umap.umap_.UMAP
                categorical data embedding

            self.umap_combine_method: str
                method to combine the embeddings
                (intersection/union/contrast/intersection_union_mapper)

            self.n_neighbors_intersection_union: int
                if umap_combine_method = intersection_union_mapper,
                number of components UMAP

            self.n_components: int
                if umap_combine_method = intersection_union_mapper,
                number of components UMAP

            self.random_state: int
                seed

            self.preprocessed_num: scipy.sparse.csr.csr_matrix
                if umap_combine_method = intersection_union_mapper,
                matrix with preprocessed numerical data

        Returns
        -------
            self.umap_combined : umap.umap_.UMAP
                combined umap

        """
        logger.info(f"Numerical data embedding shape: {self.umap_numerical_.embedding_.shape}")
        logger.info(f"Categorical data embedding shape: {self.umap_categorical_.embedding_.shape}")
   
        if self.umap_combine_method == "intersection":
            umap_combined = self.umap_numerical_ * self.umap_categorical_

        elif self.umap_combine_method == "union":
            umap_combined = self.umap_numerical_ + self.umap_categorical_

        elif self.umap_combine_method == "contrast":
            umap_combined = self.umap_numerical_ - self.umap_categorical_
        
        elif self.umap_combine_method == "intersection_union_mapper":
            intersection_mapper = umap.UMAP(
                random_state = self.random_state,
                n_neighbors = self.n_neighbors_intersection_union,
                n_components = self.n_components,
                min_dist = 0.0,
            ).fit(self.preprocessed_numerical_)

            umap_combined = intersection_mapper * (
                self.umap_numerical_ + self.umap_categorical_
            )

        else:
            raise KeyError("Select valid  UMAP combine method")

        self.umap_combined = umap_combined
        return self


    def _fit_hdbscan(self):
        """
        Fits a hdbscan model to the embedding
        
        Parameters
        ----------
            self.min_cluster_size : int
                min_cluster_size of the hdbscan model

            self.min_samples: int
                min_samples of the hdbscan model

            self.umap_combined: umap.umap_.UMAP
                combined umap embedding (numerical + categorical)

        Returns
        -------
            self.hdbscan_: hdbscan.hdbscan_.HDBSCAN
                hdbscan model
        """
        hdb_model = hdbscan.HDBSCAN(min_cluster_size = self.min_cluster_size,
                                    min_samples = self.min_samples,
                                    gen_min_span_tree = True).fit(self.umap_combined.embedding_)

        self.hdbscan_ = hdb_model

        return self


#Visualization
    def plot_condensed_tree(self):
        """
        Plots the condensed tree of the model
        
        Parameters
        ----------
            self.hdb_model:
                hdbscan model

        Returns
        -------
            None

        """

        plt.figure(figsize = (10, 8), dpi = 80)

        _ = self.hdbscan_.condensed_tree_.plot(
        select_clusters = True,
        selection_palette = sns.color_palette("deep", np.unique(self.hdbscan_.labels_).shape[0]),
    )


    def plot_embedding_labels(self):
        """
        Plots a jointplot with the model's labels
        
        Parameters
        ----------
            self.hdb_model:
                hdbscan model

            self.umap_embedding:
                data's umap embedding

        Returns
        -------
            None

        """
        if self.umap_combined.embedding_.shape[1] > 1:
            _ = sns.jointplot(
            x = self.umap_combined.embedding_[:, 0],
            y = self.umap_combined.embedding_[:, 1],
            hue = self.hdbscan_.labels_,
            kind = "kde",
            palette = "viridis"
    )
        else:
            print("The embedding has only 1 dimension, increase it to plot the results")

    def plot_2d_labels(self):
        if self.umap_combined.embedding_.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            unique_labels = np.unique(self.hdbscan_.labels_)

            #colors = ["red", "blue", "green", "yellow"]
            colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask_label = self.hdbscan_.labels_ == label
                ax.scatter(self.umap_combined.embedding_[:,0][mask_label],
                                    self.umap_combined.embedding_[:,1][mask_label],
                                    color=colors[i],
                                    alpha=0.5,
                                    s=6,
                                    label=label)

            ax.legend(fontsize=8, markerscale=1)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig
            
        else:
            print("This function works only with 2d embeddings")



#data visualization
    def assing_results(self, data):
        """
        Assings hdb_model's labels to the original data
        
        Parameters
        ----------
            data : pandas.DataFrame
                Original pandas DataFrame

        Returns
        -------
            results: pandas.DataFrame
                new pandas dataframe with the calculated clusters

        """
        results = data.copy()
        results["Cluster"] = self.hdbscan_.labels_

        return results

    def cluster_summary(self, results_df, metric = "mean", include_cat = False):
        """
        Creates a cluster's summary of the numerical and/or categorical features

        Parameters
        ----------
            results_df : pandas.DataFrame
                pandas dataframe with a cluster column

            metric: str, default = "mean"
                metric to use in the summary (mean/median/max/min)

            include_cat: bool, default = False
                include the mode of the categorical variables

        Returns
        -------
            df_summary: pandas.DataFrame
                New dataframe with the summary

        """
        #Profile by cluster
        numerics = results_df.select_dtypes(include = [int, float]).drop(["Cluster"], 1).columns.tolist()

        prop_cluster = results_df["Cluster"].value_counts(dropna = False, normalize = True).to_frame().rename(columns = {"Cluster": "data_prop"})
        count_cluster = results_df["Cluster"].value_counts(dropna = False, normalize = False).to_frame().rename(columns = {"Cluster": "data_count"})

        if metric == "mean":
            summary_data = results_df[numerics + ["Cluster"]].groupby(["Cluster"]).mean()
        elif metric == "median":
            summary_data = results_df[numerics + ["Cluster"]].groupby(["Cluster"]).median()
        elif metric == "max":
            summary_data = results_df[numerics + ["Cluster"]].groupby(["Cluster"]).max()
        elif metric == "min":
            summary_data = results_df[numerics + ["Cluster"]].groupby(["Cluster"]).min()  
        else:
            print("Select a valid metric: mean, median, max, min")
            return None

        #merge all the results
        df_summary = prop_cluster.merge(count_cluster, left_index = True, right_index = True).merge(summary_data, left_index = True, right_index = True)

        if include_cat:
            #Calculates the mode of the categorical variables
            categoricals_col = results_df.select_dtypes(exclude = ["float", "int", "datetime"]).columns.tolist()
            df_summary_cat = results_df[categoricals_col + ["Cluster"]].groupby(["Cluster"]).agg(pd.Series.mode)
            df_summary = df_summary.merge(df_summary_cat, left_index = True, right_index = True)

        return df_summary


    def describe_cluster(self, results_df, clusters = [0], columns_analyze_numerical = [],
                    columns_analyze_categorical = [], metric = "mean"):
        """
        Describes the selected clusters

        Parameters
        ----------
            results_df : pandas.DataFrame
                pandas dataframe with a cluster column

            clusters: list
                list with clusters to describe (int)

            columns_analyze_numerical: list
                list of numerical columns to describe

            columns_analyze_categorical: list
                list of categorical columns to describe

            metric: str, default = "mean"
                metric to use in the summary of numerical columns (mean/median)


        Returns
        -------
            None: None

        """

        total_rows = results_df.shape[0]

        #Iterate over the selected_clusters
        for selected_cluster in clusters:

            cluster_results = results_df[results_df["Cluster"] == selected_cluster]

            print(f"Analysis of cluster {selected_cluster}:")

            rows_cluster = cluster_results.shape[0]
            percentage_cluster = (rows_cluster/total_rows) * 100
            print(f"The cluster {selected_cluster} has {rows_cluster} rows ({percentage_cluster:.2f}% of total).")

            #Analyze the numerical columns if exists
            if len(columns_analyze_numerical) > 0:
                if metric == "mean":
                    for n_col in columns_analyze_numerical:
                        col_mean = results_df[n_col].mean()
                        col_cluster = cluster_results[n_col].mean()

                        variation = (col_cluster - col_mean)/col_mean

                        print(f"The average {n_col} in the dataset is {col_mean:.2f} and in the cluster {selected_cluster} is {col_cluster:.2f} ({variation * 100:+.1f}%).")

                elif metric == "median":
                    for n_col in columns_analyze_numerical:
                        col_median = results_df[n_col].median()
                        col_cluster = cluster_results[n_col].median()

                        variation = (col_cluster - col_median)/col_median

                        print(f"The median {n_col} in the dataset is {col_median:.2f} and in the cluster {selected_cluster} is {col_cluster:.2f} ({variation * 100:+.1f}%).")

                else:
                    print("Select a valid metric (mean/median)")

            #Analyze the categorical columns if exists
            if len(columns_analyze_categorical) > 0:
                for n_col in columns_analyze_categorical:
                    col_mode = results_df[n_col].value_counts(normalize = True, dropna = False).sort_values(ascending = False)
                    col_mode_cat, col_mode_value = col_mode.index[0], col_mode[0] * 100

                    col_cluster = cluster_results[n_col].value_counts(normalize = True, dropna = False).sort_values(ascending = False)
                    clus_mode_cat, clus_mode_value = col_cluster.index[0], col_cluster[0] * 100


                    print(f"The most common value of the column {n_col} in the dataset is {col_mode_cat} ({col_mode_value:.1f}%) and in the cluster {selected_cluster} is {clus_mode_cat} ({clus_mode_value:.1f}%).")



            print("\n")

        return None   

#Optimize model
    def tune_model(self,
                    n_trials = 100, min_cluster_start = 0.01, min_cluster_end = 0.15,
                    min_samples_start = 0.01, min_samples_end = 0.15, max_epsilon = None):
        """
        Tunes a hdbscan model maximizing the DBCV score (https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf)

        Parameters
        ----------
            n_trials : int, default = 100
                number of iterations

            min_cluster_start: float, default = 0.01
                lowest value of min_cluster of the search space (proportion of data)

            min_cluster_end: float, default = 0.15
                highest value of min_cluster of the search space (proportion of data)

            min_samples_start: float, default = 0.01
                lowest value of min_samples of the search space (proportion of data)

            min_samples_end: float, default = 0.15
                highest value of min_samples of the search space (proportion of data)

            max_epsilon: float, default = None
                If a value is provided, an optimal epsilon is searched
                between 0 and max_epsilon

        Returns
        -------
            None:
                optimized hdbscan
        """

        n_rows = self.umap_combined.embedding_.shape[0]

        #Check values:
        assert 0 <= min_cluster_start <= 1, "min_cluster_start must be between 0 and 1"
        assert 0 <= min_cluster_end <= 1, "min_cluster_end must be between 0 and 1"
        assert 0 <= min_samples_start <= 1, "min_samples_start must be between 0 and 1"
        assert 0 <= min_samples_end <= 1, "min_samples_end must be between 0 and 1"


        min_cluster_start_value = n_rows * min_cluster_start
        min_cluster_end_value = n_rows * min_cluster_end

        min_samples_start_value = n_rows * min_samples_start
        min_samples_end_value = n_rows * min_samples_end

        # 1. Define an objective function to be maximized.
        def objective(trial):

            # 2. Suggest values for the hyperparameters using a trial object.
            min_cluster_number = trial.suggest_int("min_cluster", min_cluster_start_value, min_cluster_end_value, log = False)
            min_samples_number = trial.suggest_int("min_samples", min_samples_start_value, min_samples_end_value, log = False)
            
            if max_epsilon is not None:
                cluster_selection_epsilon_number = trial.suggest_float("cluster_selection_epsilon", 0, max_epsilon)
                hdb_model = hdbscan.HDBSCAN(min_cluster_size = min_cluster_number,
                        min_samples = min_samples_number,
                        cluster_selection_epsilon = cluster_selection_epsilon_number,
                        gen_min_span_tree = True).fit(self.umap_combined.embedding_)
            else:
                hdb_model = hdbscan.HDBSCAN(min_cluster_size = min_cluster_number,
                        min_samples = min_samples_number,
                        gen_min_span_tree = True).fit(self.umap_combined.embedding_)

            score = hdb_model.relative_validity_

            return score

        # 3. Create a study object and optimize the objective function.
        study = optuna.create_study(direction = 'maximize')
        study.optimize(objective, n_trials = n_trials)

        #Train the model with the optimized parameters
        best_model_params = study.best_params

        print("Best parameters: ", best_model_params)

        if max_epsilon is not None:
            hdb_model = hdbscan.HDBSCAN(min_cluster_size = best_model_params['min_cluster'],
                        min_samples = best_model_params['min_samples'],
                        cluster_selection_epsilon = best_model_params['cluster_selection_epsilon'],
                        gen_min_span_tree = True).fit(self.umap_combined.embedding_)
        else:
            hdb_model = hdbscan.HDBSCAN(min_cluster_size = best_model_params['min_cluster'],
                min_samples = best_model_params['min_samples'],
                gen_min_span_tree = True).fit(self.umap_combined.embedding_)

        self.hdbscan_ = hdb_model

        return self