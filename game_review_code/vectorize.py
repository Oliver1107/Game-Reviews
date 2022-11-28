"""Vectorizing summary column of reviews."""


import pandas as pd
import spacy
from collections import Counter


def tokenizer(text, nlp):
    token_list = []
    doc = nlp(text)
    for token in doc:
        if not any([token.is_punct, token.is_stop, (token.pos_ == 'PRON')]):
            token_list.append(token.lemma_.lower())
    return token_list


def count(token_lists):
    doc_count = Counter()
    for token_list in token_lists:
        doc_count.update(set(token_list))
    doc_count_dict = zip(doc_count.items(), doc_count.values())
    dc = pd.DataFrame(doc_count_dict, columns=['word', 'appears_in_docs'])
    return dc


def trimmer(tokens, trim):
    trimmed = []
    for token in tokens:
        if token not in trim:
            trimmed.append(token)
    return trimmed


if __name__ == '__main__':
    df = pd.read_csv('game_data/wrangled_reviews.csv').drop(
        columns=['Unnamed: 0'])
    nlp = spacy.load('nlp_model')

    token_lists = [tokenizer(summary, nlp) for summary in df['Summary']]

    wc = count(token_lists)

    max = len(df) * 0.25
    min = len(df) * 0.05

    trim = wc['word'][(wc['appears_in_docs'] > max) |
                      (wc['appears_in_docs'] < min)]
    trim = trim.to_list()

    df['tokens'] = df['Summary'].apply(lambda text: tokenizer(text, nlp))
    df['tokens'] = df['tokens'].apply(lambda tokens: trimmer(tokens, trim))
    drop = df[df['tokens'].apply(len) < 20].index
    df.drop(drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['tokens'] = df['tokens'].apply(' '.join)
    vectors = df['tokens'].apply(lambda x: nlp(x).vector)
    split = pd.DataFrame(vectors.to_list())
    df = pd.concat([df, split], axis=1).drop(columns=['tokens'])

    df.to_csv('game_data/vectored_reviews.csv')
    pd.to_pickle(trim, 'game_review_code/trim_list.pkl')
