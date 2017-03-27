import os
import pandas as pd
from sklearn.externals import joblib
# from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    df_pickle_filename = 'df_music.pkl'
    df_pickle_path = os.path.join('/data/music', df_pickle_filename)
    df = pd.read_pickle(df_pickle_path)

    # shuffle rows for cross-validation
    df = df.sample(frac=1)

    y = df.pop('genre').values
    del df['movie']
    X = df.values


    final_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_model.pkl')
    clf = joblib.load(final_model_path)
    print clf.predict(X)
