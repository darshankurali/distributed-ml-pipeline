from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

class DistributedMLPipeline:
    """Distributed training with hyperparameter optimization"""
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs  # -1 for all cores
        self.best_models = {}
        self.search_results = {}
    
    def create_pipeline(self, clf_name='random_forest'):
        """Create ML pipeline"""
        if clf_name == 'random_forest':
            clf = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        else:
            clf = GradientBoostingClassifier(random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        return pipeline
    
    def hyperparameter_grid(self, model_type='random_forest'):
        """Define hyperparameter search space"""
        if model_type == 'random_forest':
            return {
                'classifier__n_estimators': [50, 100, 200, 300],
                'classifier__max_depth': [5, 10, 15, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        else:
            return {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
    
    def optimize_hyperparameters(self, X, y, model_type='random_forest', cv=5, search_type='random'):
        """Grid/Random search for best hyperparameters"""
        pipeline = self.create_pipeline(model_type)
        param_grid = self.hyperparameter_grid(model_type)
        
        if search_type == 'grid':
            search = GridSearchCV(
                pipeline, param_grid, cv=cv, n_jobs=self.n_jobs,
                verbose=2, scoring='roc_auc'
            )
        else:
            search = RandomizedSearchCV(
                pipeline, param_grid, n_iter=20, cv=cv,
                n_jobs=self.n_jobs, verbose=2, random_state=42,
                scoring='roc_auc'
            )
        
        print(f"Starting {search_type.upper()} search with {self.n_jobs} jobs...")
        search.fit(X, y)
        
        self.best_models[model_type] = search.best_estimator_
        self.search_results[model_type] = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
        
        print(f"\nBest params: {search.best_params_}")
        print(f"Best CV Score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def compare_models(self, X, y):
        """Train and compare multiple models in parallel"""
        models = ['random_forest', 'gradient_boosting']
        
        # Parallel model training
        trained_models = Parallel(n_jobs=self.n_jobs)(
            delayed(self.optimize_hyperparameters)(X, y, model, cv=5, search_type='random')
            for model in models
        )
        
        return trained_models
    
    def get_best_model(self):
        """Return best model across all"""
        best = max(
            self.search_results.items(),
            key=lambda x: x[1]['best_score']
        )
        
        return {
            'model_type': best[0],
            'model': self.best_models[best[0]],
            'score': best[1]['best_score']
        }

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    print("Loading data...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Starting distributed hyperparameter optimization...")
    pipeline = DistributedMLPipeline(n_jobs=-1)
    
    # Optimize RF
    print("\n1. Optimizing Random Forest...")
    rf_model = pipeline.optimize_hyperparameters(X_train, y_train, 'random_forest', search_type='random')
    
    # Optimize GB
    print("\n2. Optimizing Gradient Boosting...")
    gb_model = pipeline.optimize_hyperparameters(X_train, y_train, 'gradient_boosting', search_type='random')
    
    # Get best
    print("\n3. Results Summary:")
    best = pipeline.get_best_model()
    print(f"Best Model: {best['model_type']} (Score: {best['score']:.4f})")
