"""Generate predictions using model."""


import pandas as pd
import game_review_code.model_train
import __main__


df = pd.read_csv('game_data/wrangled_reviews.csv').drop(columns=['Unnamed: 0'])

ratings = df['Rating'].unique()
developers = df['Developer'].unique()
players = df['Number of Players'].unique()
genres = df.columns[12:]
cols = list(df.drop(columns=[
    'Metascore', 'User Score', 'Platform',
    'Release Day of Month', 'Title']).columns)

__main__.Vectorizer = game_review_code.model_train.Vectorizer

model_ms = pd.read_pickle('ml_models/ms_score.pkl')
model_us = pd.read_pickle('ml_models/us_score.pkl')


def prediction(summary, rating, developer, players, online,
               month, year, genre):
    genre = [int(g in genre) for g in genres]
    game = [summary, rating, developer, players, online, month, year]
    game.extend(genre)
    game = pd.DataFrame(columns=cols, data=[game])
    ms = model_ms.predict(game)[0]
    us = model_us.predict(game)[0]
    return [ms, us]
