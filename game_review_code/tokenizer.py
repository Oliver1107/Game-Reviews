"""Tokenize summary column of reviews."""


import pandas as pd
from collections import Counter


def tokenizer(text, nlp):
    """Tokenize text and return list of lemmas."""
    token_list = []
    doc = nlp(text)
    for token in doc:
        if not any([token.is_punct, token.is_stop, (token.pos_ == 'PRON')]):
            token_list.append(token.lemma_.lower())
    return token_list


def word_count(token_lists):
    """Create a dataframe of word counts using lists of tokens."""
    doc_count = Counter()
    for token_list in token_lists:
        doc_count.update(set(token_list))
    doc_count_dict = zip(doc_count.items(), doc_count.values())
    dc = pd.DataFrame(doc_count_dict, columns=['word', 'appears_in_docs'])
    dc['word'] = dc['word'].apply(lambda word: word[0])
    return dc
