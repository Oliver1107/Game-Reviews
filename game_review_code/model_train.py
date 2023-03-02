"""Train predictor model."""


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

warnings.filterwarnings('ignore')


# create a vectorizer model to use in pipeline
class Vectorizer:
    """Transform text columns into vector columns."""
    def __init__(self, max_df=0.99, min_df=0.01, n_components=0):
        self.max = max_df
        self.min = min_df
        self.n_comp = n_components

    def set_params(self, max_df=0.99, min_df=0.01, n_components=0):
        self.max = max_df
        self.min = min_df
        self.n_comp = n_components

    def fit(self, X, y=None):
        corpus = X['Summary']
        self.vect = TfidfVectorizer(stop_words='english',
                                    ngram_range=(1, 2),
                                    max_df=self.max,
                                    min_df=self.min)
        self.vect.fit(corpus)
        X_vect = self.vect.transform(corpus)
        vectors = pd.DataFrame(X_vect.todense())
        if self.n_comp:
            X_vect = self.vect.transform(corpus)
            vectors = pd.DataFrame(X_vect.todense())
            self.svd = TruncatedSVD(n_components=self.n_comp,
                                    algorithm='randomized',
                                    n_iter=10)
            self.svd.fit(vectors)

    def transform(self, X):
        corpus = X['Summary']
        X_vect = self.vect.transform(corpus)
        vectors = pd.DataFrame(X_vect.todense(), index=X.index)
        if self.n_comp:
            X_svd = self.svd.transform(vectors)
            vectors = pd.DataFrame(X_svd, index=X.index)
        vectors.columns = vectors.columns.astype(str)
        return pd.concat([X, vectors], axis=1).drop(columns=['Summary'])

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __repr__(self):
        return f"Vectorizer(max={self.max}, " \
               f"min={self.min}, n_comp={self.n_comp})"


if __name__ == '__main__':
    # read in vectored reviews data
    df = pd.read_csv('game_data/wrangled_reviews.csv').drop(
        columns=['Unnamed: 0'])

    # create feature matrix without non-helpful columns and target vectors
    X = df.drop(columns=[
        'Metascore', 'User Score', 'Title', 'Platform',
        'Release Day of Month'])
    y = (df[['Metascore', 'User Score']])

    # create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8,  random_state=42)

    y1_train = y_train['Metascore']
    y2_train = y_train['User Score']
    y1_test = y_test['Metascore']
    y2_test = y_test['User Score']

    # instantiate model and parameters for parameter search
    model = make_pipeline(Vectorizer(),
                          OneHotEncoder(),
                          StandardScaler(),
                          KNeighborsClassifier(n_jobs=-1))

    parameters_ms = {'vectorizer__max_df': [0.5],
                     'vectorizer__min_df': [5, 10, 15],
                     'vectorizer__n_components': [1, 3, 5],
                     'kneighborsclassifier__p': [1, 2],
                     'kneighborsclassifier__weights': ['distance']}
    parameters_us = {'vectorizer__max_df': [0.5],
                     'vectorizer__min_df': [5, 10, 15],
                     'vectorizer__n_components': [1, 3, 5],
                     'kneighborsclassifier__p': [1, 2],
                     'kneighborsclassifier__weights': ['distance']}

    # search parameters for best model for metascore and user score
    search_ms = GridSearchCV(model, parameters_ms, cv=5)
    search_ms.fit(X_train, y1_train)
    model_ms = search_ms.best_estimator_
    model_ms.fit(X_train, y1_train)

    search_us = GridSearchCV(model, parameters_us, cv=5)
    search_us.fit(X_train, y2_train)
    model_us = search_us.best_estimator_
    model_us.fit(X_train, y2_train)

    # save models to pickle file
    pd.to_pickle(model_ms, 'ml_models/ms_score.pkl')
    pd.to_pickle(model_us, 'ml_models/us_score.pkl')
