"""Generate predictions using model."""


import pandas as pd
import spacy
from game_review_code.vectorize import tokenizer, trimmer


df = pd.read_csv('game_data/vectored_reviews.csv').drop(
    columns=['Unnamed: 0', 'Summary'])

ratings = df['Rating'].unique()
developers = df['Developer'].unique()
players = df['Number of Players'].unique()
genres = df.columns[11:-96]
cols = list(df.drop(columns=[
    'Metascore', 'User Score', 'Platform',
    'Release Day of Month', 'Title']).columns)

model_ms = pd.read_pickle('ml_models/ms_score.pkl')
model_us = pd.read_pickle('ml_models/us_score.pkl')
nlp = spacy.load('nlp_model')
trim = pd.read_pickle('game_review_code/trim_list.pkl')


def prediction(rating, developer, players, online,
               month, year, genre, summary):
    genre = [int(g in genre) for g in genres]
    tokens = tokenizer(summary, nlp)
    trimmed = trimmer(tokens, trim)
    summary = ' '.join(trimmed)
    summary = nlp(summary).vector
    game = [rating, developer, players, online, month, year]
    game.extend(genre)
    game.extend(summary)
    game = pd.DataFrame(columns=cols, data=[game])
    ms = round(model_ms.predict(game)[0])
    us = round(model_us.predict(game)[0], 1)
    if ms > 99:
        ms = 99
    elif ms < 1:
        ms = 1
    if us > 99:
        us = 99
    elif us < 1:
        us = 1
    if len(trimmed) < 10:
        return [ms, us, 'warning']
    return [ms, us]
