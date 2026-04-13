# Project 14: Distributed ML with Hyperparameter Tuning

## 🎯 Overview
Parallel hyperparameter optimization using GridSearchCV and RandomizedSearchCV across CPU cores.

**Industry Application**: AutoML, model selection at scale

## 🏗️ Architecture
```
Training Data
       ↓
Parameter Grid Definition
       ↓
┌──────────────────────────────┐
│ Core-1: HP Set 1 → CV Train  │
│ Core-2: HP Set 2 → CV Train  │
│ Core-N: HP Set N → CV Train  │
└──────────────────────────────┘
       ↓
Results Aggregation
       ↓
Best Parameters + Best Model
```

## 🚀 Setup

```bash
pip install -r requirements.txt

python src/distributed_pipeline.py
```

## 📊 Features
- **Parallel CV**: Use all CPU cores
- **Grid Search**: Exhaustive search
- **Random Search**: Fast approximate search
- **Model Comparison**: Multiple algorithms
- **Cross-Validation**: Robust evaluation

## 💼 Production Benefits
- **Time**: 8x faster with 8 cores
- **Accuracy**: Find optimal hyperparameters
- **Reproducibility**: Documented best params

## 🎓 Skills Demonstrated
- Scikit-learn pipelines
- Grid/Random search
- Parallel processing (Joblib)
- Cross-validation
- Model evaluation
