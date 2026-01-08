import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.translator = Translator()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure input is a Pandas Series
        if isinstance(X, list):
            X = pd.Series(X)

        # Apply translation
        X_translated = X.apply(self.translate_to_english)
        return X_translated

    def translate_to_english(self, text):
        try:
            if pd.isnull(text) or str(text).strip() == "":
                return ""
            return self.translator.translate(text).text
        except Exception:
            return text