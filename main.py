import pandas as pd
import numpy as np
from tqdm import tqdm
from variables import *
from utils import *
from sklearn.decomposition import PCA

def run(X, y, model, model_name, param_grid, scoring='f1', cv=10):
    
    results = pd.DataFrame(columns=["Technique", "Model", "F1 Score", "Parameters", "Fold"])
    
    for technique in tqdm(TECHNIQUES, desc="Resampling Techniques"):
        X_resampled, y_resampled = apply_sampling_technique(X, y, technique)

        cv_results = apply_grid_search_cv(
            X_resampled, y_resampled, model, param_grid, scoring=scoring, cv=cv
        )
        
        for i in tqdm(range(len(cv_results['mean_test_score'])), desc="CV Iterations", leave=False):
            new_row = pd.DataFrame([{
                "Technique": technique,
                "Model": model_name,
                "F1 Score": cv_results['mean_test_score'][i],
                "Parameters": cv_results['params'][i],
                "Fold": i + 1
            }])
            
            results = pd.concat([results, new_row], ignore_index=True)
        
    return results

if __name__ == "__main__":
    
    X, y = load_and_preprocess_data()
    
    for model_name, model in MODELS.items():
        
        results = run(
            X=X, 
            y=y, 
            model=model, 
            param_grid=PARAMETERS[model_name], 
            model_name=model_name,
            scoring='f1',
            cv=10
        )
        
        # Save results to CSV
        results.to_csv(f"results/{model_name}.csv", index=False)
        
        # Print results
        print(f"Results for {model_name} saved to {model_name}_results.csv")
