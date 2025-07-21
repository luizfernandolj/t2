import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss


def load_and_preprocess_data(path='train.csv'):
    """
    Lê e pré-processa os dados de aplicação de empréstimo a partir do arquivo 'train.csv'.
    Esta função executa os seguintes passos:
        - Carrega o conjunto de dados a partir do arquivo 'train.csv'.
        - Renomeia a coluna 'LoanApproved' para 'class' e remove a coluna original 'LoanApproved'.
        - Remove quaisquer linhas que contenham valores ausentes.
        - Converte a coluna 'ApplicationDate' de string para datetime, e então transforma as datas em valores ordinais.
        - Separa o conjunto de dados em features (X) e alvo (y), onde y corresponde à coluna 'class'.
        - Identifica as colunas categóricas e numéricas nas features.
        - Constrói um ColumnTransformer que:
            • Aplica OneHotEncoder (ignorando categorias desconhecidas) nas colunas categóricas.
            • Aplica StandardScaler nas colunas numéricas.
            • Utiliza 'passthrough' para quaisquer colunas não explicitamente transformadas.
        - Utiliza todos os núcleos de CPU disponíveis para processamento através do parâmetro n_jobs=-1 no ColumnTransformer.
    Retorna:
        tuple: Uma tupla (X, y) onde:
            - X é a matriz de features pré-processada (como numpy array ou matriz esparsa, dependendo dos transformers),
            - y é o vetor alvo extraído da coluna 'class'.
    """
    df = pd.read_csv(path)

    df["class"] = df["LoanApproved"]
    df.drop("LoanApproved", axis=1, inplace=True)
    df.dropna(inplace=True)

    df["ApplicationDate"] = pd.to_datetime(df["ApplicationDate"])
    df["ApplicationDate"] = df["ApplicationDate"].apply(lambda x: x.toordinal())


    # Separar features (X) e target (y)
    X = df.drop("class", axis=1)
    y = df["class"]

    # Identificar colunas categóricas e numéricas
    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(exclude='object').columns

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
    X = preprocessor.fit_transform(X)
    
    
    return X, y

def apply_grid_search_cv(X, y, estimator, param_grid, scoring='f1', cv=10):
    """
    Aplica GridSearchCV para encontrar os melhores hiperparâmetros para o estimador fornecido.
    
    Parâmetros:
        X (array-like): Matriz de características.
        y (array-like): Vetor de rótulos.
        estimator: Estimador a ser ajustado (ex: RandomForestClassifier).
        param_grid (dict): Dicionário com os parâmetros a serem ajustados.
        scoring (str): Métrica de avaliação a ser usada (padrão é 'f1').
        cv (int): Número de folds para validação cruzada (padrão é 5).
        
    Retorna:
        dict: Resultados do GridSearchCV.
    """
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,     # Usando F1-score para avaliação
        cv=cv,         # Mostra o progresso detalhado
        n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search.cv_results_

def apply_sampling_technique(X, y, technique, **kwargs):
    """
    Aplica uma técnica de amostragem (undersampling, oversampling, SMOTE, etc.) nos dados.
    
    Parâmetros:
        X (array-like): Matriz de características.
        y (array-like): Vetor de rótulos.
        technique (str): Nome da técnica de amostragem a ser aplicada ('undersample', 'oversample', 'smote', 'tomeklinks', 'nearmiss').
        **kwargs: Argumentos adicionais específicos para cada técnica.
        
    Retorna:
        tuple: Dados transformados (X_resampled, y_resampled).
    """
    if technique == 'RandomUnderSampler':
        sampler = RandomUnderSampler(**kwargs)
    elif technique == 'RandomOverSampler':
        sampler = RandomOverSampler(**kwargs)
    elif technique == 'SMOTE':
        sampler = SMOTE(**kwargs)
    elif technique == 'TomekLinks':
        sampler = TomekLinks(**kwargs)
    elif technique == 'NearMiss':
        sampler = NearMiss(**kwargs)
    else:
        raise ValueError("Técnica de amostragem desconhecida: {}".format(technique))
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def evaluate_f1_scores(X, y, classifier, cv=10):
    """
    Avalia o F1-score de um classificador usando validação cruzada.
    
    Parâmetros:
        X (array-like): Matriz de características.
        y (array-like): Vetor de rótulos.
        classifier: Classificador a ser avaliado (ex: RandomForestClassifier).
        scoring (str): Métrica de avaliação a ser usada (padrão é 'f1').
        cv (int): Número de folds para validação cruzada (padrão é 5).
        
    Retorna:
        list: Lista de F1-scores para cada fold.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    
    return f1_scores

def get_best_estimator(grid_search_results):
    """
    Obtém o melhor estimador a partir dos resultados do GridSearchCV utilizando a métrica F1.
    
    Parâmetros:
        grid_search_results (dict): Resultados do GridSearchCV, onde a chave "mean_test_score" corresponde ao F1 score.
        
    Retorna:
        dict: Dicionário com os melhores parâmetros e a melhor pontuação.
    """
    best_index = np.argmax(grid_search_results["mean_test_score"])
    best_params = grid_search_results["params"][best_index]
    best_score = grid_search_results["mean_test_score"][best_index]
    
    return {
        'best_params': best_params,
        'best_score': best_score
    }
