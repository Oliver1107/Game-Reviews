"""Train predictor model."""


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score

df = pd.read_csv('game_data/vectored_reviews.csv').drop(columns=['Unnamed: 0'])

X = df.drop(columns=[
    'Metascore', 'User Score', 'Title', 'Platform',
    'Summary', 'Release Day of Month'])
y = df[['Metascore', 'User Score']]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

y1_train = y_train['Metascore']
y2_train = y_train['User Score']
y1_test = y_test['Metascore']
y2_test = y_test['User Score']


model_r = make_pipeline(OneHotEncoder(),
                        StandardScaler(),
                        Ridge(random_state=42))

score = make_scorer(r2_score)
parameters_r = {'ridge__alpha': range(1250, 1401, 5)}

search_ms_r = GridSearchCV(
    model_r, parameters_r, cv=5, scoring=score)
search_ms_r.fit(X_train, y1_train)
model_ms_r_ = search_ms_r.best_estimator_
model_ms_r_.fit(X_train, y1_train)

search_us_r = GridSearchCV(
    model_r, parameters_r, cv=5, scoring=score)
search_us_r.fit(X_train, y2_train)
model_us_r_ = search_us_r.best_estimator_
model_us_r_.fit(X_train, y2_train)


pd.to_pickle(model_ms_r_, 'ml_models/ms_score.pkl')
pd.to_pickle(model_us_r_, 'ml_models/us_score.pkl')
