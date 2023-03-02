"""Game review web application."""


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder
from flask import Flask, render_template, request
from game_review_code.model_train import Vectorizer
from game_review_code.predict_score import (prediction, genres, ratings,
                                            players, developers)


def create_app():
    """
    Creates app and routes.
    """

    app = Flask(__name__)

    @app.route('/')
    def root():
        return render_template('base.html')

    @app.route('/game-stats')
    def stats():
        return render_template('game_stats.html')

    @app.route('/predict-score', methods=['POST', 'GET'])
    def predict():
        if request.method == 'GET':
            ms_score = 'No score yet.'
            us_score = 'No score yet.'

        if request.method == 'POST':
            try:
                rating = request.values['rating']
                developer = request.values['developer']
                player = request.values['player']
                online = int(request.values['online'])
                month = int(request.values['month'])
                year = int(request.values['year'])
                genre = []
                for val in request.values:
                    if val in genres:
                        genre.append(val)
                summary = request.values['summary']

                preds = prediction(summary, rating, developer,
                                   player, online, month,
                                   year, genre)
                ms_score = preds[0]
                us_score = preds[1]

            except ValueError:
                ms_score = 'Not enough data.'
                us_score = 'Not enough data.'

        return render_template(
            'predict_score.html', ratings=ratings, developers=developers,
            players=players, genres=genres, ms_score=ms_score,
            us_score=us_score)

    df = pd.read_csv('game_data/wrangled_reviews.csv').drop(
        columns=['Unnamed: 0'])
    drop = df[df['Developer'] == 'Other'].index
    df.drop(drop, inplace=True)

    memory = {}

    @app.route('/recommend-games', methods=['GET', 'POST'])
    def recommend(memory=memory):
        if request.method == 'GET':
            memory.clear()

            sample = df.sample(1)
            index = sample.index[0]
            title = sample['Title'].iloc[0]
            dev = sample['Developer'].iloc[0]
            month = sample['Release Month'].iloc[0]
            day = sample['Release Day of Month'].iloc[0]
            year = sample['Release Year'].iloc[0]
            summary = sample['Summary'].iloc[0]

        if request.method == 'POST':
            game = int(request.values['index'])
            val = int(request.values['val'])
            memory[game] = val
            sample = df.sample(1)
            while sample.index[0] in memory:
                sample = df.sample(1)

            index = sample.index[0]
            title = sample['Title'].iloc[0]
            dev = sample['Developer'].iloc[0]
            month = sample['Release Month'].iloc[0]
            day = sample['Release Day of Month'].iloc[0]
            year = sample['Release Year'].iloc[0]
            summary = sample['Summary'].iloc[0]

        return render_template('recommendations.html', title=title,
                               dev=dev, month=month, day=day, year=year,
                               summary=summary, index=index, memory=memory)

    @app.route('/show-recommendations', methods=['POST'])
    def show_recs():
        try:
            memory = eval(request.values['memory'])
            vect_df = Vectorizer(max_df=0.25, min_df=0.05).fit_transform(df)
            vect_df['Summary'] = df['Summary']
            samp_df = vect_df.loc[list(memory.keys())]
            samp_df['target'] = list(memory.values())
            X = samp_df.drop(
                columns=['Title', 'Platform',
                         'Release Day of Month', 'target'])
            y = samp_df['target']
            model = make_pipeline(OneHotEncoder(),
                                  StandardScaler(),
                                  LogisticRegression())
            model.fit(X, y)

            rdf = vect_df.drop(list(memory.keys())).drop(
                columns=['Platform', 'Release Day of Month'])
            tdf = rdf.drop(columns=['Title'])

            preds = model.predict_proba(tdf)
            thresh = 0.70
            recs = []
            for i in range(len(preds)):
                pred = preds[i]
                if pred[1] > thresh:
                    recs.append((i, pred[1]))

            recommendations = rdf.iloc[[rec[0] for rec in recs]].copy()
            recommendations['proba'] = [rec[1] for rec in recs]
            recommendations.sort_values(by='proba', ascending=False)

            num_recs = min([10, len(recommendations)])
            games = []
            for i in range(num_recs):
                genre_cols = recommendations.columns[9:-98]
                genres = []
                for col in genre_cols:
                    if recommendations.iloc[i][col]:
                        genres.append(col)
                genres = ', '.join(genres)
                title = recommendations.iloc[i]['Title']
                dev = recommendations.iloc[i]['Developer']
                summary = recommendations.iloc[i]['Summary']
                games.append(((i + 1), (title, dev, genres, summary)))

            return render_template('show_recommendations.html',
                                   games=games, num=num_recs)
        except ValueError:
            return render_template('error_rec.html')

    return app
