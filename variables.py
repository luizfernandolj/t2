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
    "KNeighborsClassifier": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
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
    "n_estimators": [10, 50, 100, 200],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}


PARAMETERS = {
    "KNeighborsClassifier": parameters_knn,
    "GaussianNB": parameters_nb,
    "DecisionTreeClassifier": parameters_dt,
    "RandomForestClassifier": parameters_rf
}