from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )