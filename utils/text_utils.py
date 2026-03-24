from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def extract_keywords(texts, top_n=20):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    X = vectorizer.fit_transform(texts)
    scores = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def build_wordcloud(text, **kwargs):
    return WordCloud(width=800, height=400, background_color='white', stopwords=set(), **kwargs).generate(text)
