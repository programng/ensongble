import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

def load_data():
    pickle_filename = 'df_music.pkl'
    pickle_path = os.path.join('/data/music', pickle_filename)
    df = pd.read_pickle(pickle_path)
    return df

def filter_genres(df, genres):
    return df[df['genre'].isin(genres)]

if __name__ == '__main__':
    # genres = ['family', 'sci-fi']
    # genres = ['horror', 'sci-fi']
    genres = ['family', 'horror']
    # genres = ['family', 'horror', 'sci-fi']
    df = load_data()
    df = filter_genres(df, genres)

    # shuffle rows for cross-validation
    df = df.sample(frac=1)

    y = df.pop('genre').values
    del df['movie']
    X = df.values
    clf = RandomForestClassifier()
    clf.fit(X, y)

    final_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_model.pkl')

    joblib.dump(clf, final_model_path)
    # clf = joblib.load(final_model_path)
