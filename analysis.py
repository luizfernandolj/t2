import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from utils import load_and_preprocess_data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def read_data(path="results/"):
    knn = pd.read_csv(path + "KNeighborsClassifier.csv")
    rf = pd.read_csv(path + "RandomForestClassifier.csv")
    nb = pd.read_csv(path + "GaussianNB.csv")
    dt = pd.read_csv(path + "DecisionTreeClassifier.csv")
    
    knn.dropna(inplace=True)
    rf.dropna(inplace=True)
    nb.dropna(inplace=True)
    dt.dropna(inplace=True)
    
    return knn, rf, nb, dt


def apply_best_model(results_train, test, test_ids, model):
    
    best_model = results_train[results_train["Model"] == model].sort_values(by="F1 Score", ascending=False).iloc[0]
    print(best_model)
    model_name = best_model['Model']
    parameters = best_model['Parameters']
    
    if model_name == 'KNeighborsClassifier':
        model = KNeighborsClassifier(**eval(parameters))
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(**eval(parameters))
    elif model_name == 'GaussianNB':
        model = GaussianNB  (**eval(parameters))
    elif model_name == 'DecisionTreeClassifier':    
        model = DecisionTreeClassifier(**eval(parameters))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(model)
    
    X, y = load_and_preprocess_data()
    
    model.fit(X, y)
    predictions = model.predict(test)
    
    results = pd.DataFrame({
        'id': test_ids,
        'LoanApproved': predictions
    })
    
    return results
    

def process_test_data(test):
    test.dropna(inplace=True)
    ids = test['id']
    test.drop('id', axis=1, inplace=True)

    test["ApplicationDate"] = pd.to_datetime(test["ApplicationDate"])
    test["ApplicationDate"] = test["ApplicationDate"].apply(lambda x: x.toordinal())

    # Identificar colunas categóricas e numéricas
    categorical_cols = test.select_dtypes(include='object').columns
    numerical_cols = test.select_dtypes(exclude='object').columns

    # Criar o ColumnTransformer
    # O remainder='passthrough' garante que as colunas numéricas que não são transformadas fiquem no dataset
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('scaler', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough',
        n_jobs=-1  # Usar todos os núcleos disponíveis
    )

    # Aplicar as transformações
    X = preprocessor.fit_transform(test)

    return X, ids




if __name__ == "__main__":
    knn, rf, nb, dt = read_data()
    
    results = pd.concat([knn, rf, nb, dt], axis=0, ignore_index=True)
    
    best = results.loc[results.groupby('Model')['F1 Score'].idxmax()]
    
    test = pd.read_csv('test.csv')
    
    test, test_ids = process_test_data(test)
    
    predictions = apply_best_model(best, 
                                   test, 
                                   test_ids= test_ids, 
                                   model="GaussianNB")

    predictions.to_csv('predictions_nb.csv', index=False)
    