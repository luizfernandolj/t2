from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier



TECHNIQUES = [
    "RandomUnderSampler",
    "RandomOverSampler",
    "SMOTE",
    "TomekLinks",
    "NearMiss"
]

MODELS = {
    "RandomForestClassifier": RandomForestClassifier(),
    # "KNeighborsClassifier": KNeighborsClassifier(),
    # "GaussianNB": GaussianNB(),
    # "DecisionTreeClassifier": DecisionTreeClassifier()
}

parameters_knn = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "metric": ["minkowski", "euclidean", "manhattan"],
    "leaf_size": [10, 20, 30, 40, 50]
}

parameters_nb = {
    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}


parameters_dt = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}


parameters_rf = {
    "n_estimators": [10, 50, 100, 200, 300, 500, 1000],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10, 15, 20, 30, 50, 100],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "max_features": ["sqrt", "log2", None, "auto"],
    "max_leaf_nodes": [None, 10, 20, 50, 100],
    "min_weight_fraction_leaf": [0.0, 0.01, 0.05, 0.1],
    "bootstrap": [True, False],
    "oob_score": [True, False],
    "n_jobs": [-1],
    "random_state": [42],
    "verbose": [0, 1],
    "warm_start": [True, False],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "ccp_alpha": [0.0, 0.01, 0.1],
    "max_samples": [None, 0.5, 0.75, 1.0]
}


PARAMETERS = {
    "KNeighborsClassifier": parameters_knn,
    "GaussianNB": parameters_nb,
    "DecisionTreeClassifier": parameters_dt,
    "RandomForestClassifier": parameters_rf
}