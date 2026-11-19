"""Fixed baseline settings; no tuning on the held-out test set."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_models(config):
    return {
        'Logistic Regression': Pipeline([
            ('scale', StandardScaler()),
            ('classifier', LogisticRegression(random_state=config['seed'], **config['logistic_regression']))
        ]),
        'Random Forest': RandomForestClassifier(random_state=config['seed'], **config['random_forest'])
    }
