import os
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data():
    pickle_filename = 'df_music.pkl'
    pickle_path = os.path.join('/data/music', pickle_filename)
    df = pd.read_pickle(pickle_path)
    return df

def filter_genres(df, genres):
    return df[df['genre'].isin(genres)]

def custom_train_test_split(df, test_proportion=0.1):
    train_movies = []
    test_movies = []
    for genre in df['genre'].unique():
        df_genre = df[df['genre'] == genre]
        movies = df_genre['movie'].unique()
        number_of_movies = len(movies)
        test_number_of_movies = math.floor(number_of_movies * test_proportion)
        train_number_of_movies = number_of_movies - test_number_of_movies

        shuffled_movies = np.random.permutation(movies)
        train_movies = np.append(train_movies, shuffled_movies[:train_number_of_movies])
        test_movies = np.append(test_movies, shuffled_movies[train_number_of_movies:])

    df_train = df[df['movie'].isin(train_movies)]
    df_test = df[df['movie'].isin(test_movies)]

    return df_train, df_test

def destructure_df(df):
    y = df.pop('genre').values
    movies = df.pop('movie').values
    X = df.values
    return y, movies, X

def ensemble_predict(clf, X, movies):
    predicted_genres_for_each_song = clf.predict(X)
    return get_movie_genre(predicted_genres_for_each_song, movies)

def get_movie_genre(genres, movies):
    predicted_genres_for_movies_dict = defaultdict(list)
    for genre, movie in zip(genres, movies):
        predicted_genres_for_movies_dict[movie].append(genre)
    for key in predicted_genres_for_movies_dict.keys():
        predicted_genres_for_movies_dict[key] = Counter(predicted_genres_for_movies_dict[key]).most_common(1)[0][0]
    return sorted(predicted_genres_for_movies_dict.items(), key=lambda pair: pair[0])

def custom_accuracy_score(ensemble_test, ensemble_predict):
    correct = 0
    incorrect = 0
    for test, predict in zip(ensemble_test, ensemble_predict):
        if test[0] == predict[0]:
            # print(test[0], 'actual:', test[1], 'predicted:', predict[1])
            if test[1] == predict[1]:
                correct += 1
            else:
                incorrect += 1
    return 1.* correct/(correct+incorrect)

def custom_cross_validate(df, k=5):
    train_accuracy_scores = []
    test_accuracy_scores = []

    movie_partitions = [[]] * k
    test_proportion = 1./k

    for genre in df['genre'].unique():
        df_genre = df[df['genre'] == genre]
        movies = df_genre['movie'].unique()
        number_of_movies = len(movies)
        partition_number_of_movies = math.floor(number_of_movies * test_proportion)
        shuffled_movies = np.random.permutation(movies)

        for partition in range(k):
            start = partition * partition_number_of_movies
            end = (partition + 1) * partition_number_of_movies
            movie_partitions[partition] = movie_partitions[partition] + list(shuffled_movies[start:end])

    for partition in range(k):
        train_movies_list = movie_partitions[:partition] + movie_partitions[partition+1:]
        train_movies = [movie for movies in train_movies_list for movie in movies]
        test_movies = movie_partitions[partition]
        df_train = df[df['movie'].isin(train_movies)]
        df_test = df[df['movie'].isin(test_movies)]

        y_train, movies_train, X_train = destructure_df(df_train)
        y_test, movies_test, X_test = destructure_df(df_test)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        ensemble_prediction = ensemble_predict(clf, X_test, movies_test)
        ensemble_test = get_movie_genre(y_test, movies_test)
        test_accuracy_scores.append(custom_accuracy_score(ensemble_test, ensemble_prediction))

        ensemble_prediction = ensemble_predict(clf, X_train, movies_train)
        ensemble_test = get_movie_genre(y_train, movies_train)
        train_accuracy_scores.append(custom_accuracy_score(ensemble_test, ensemble_prediction))

    return train_accuracy_scores, test_accuracy_scores


if __name__ == '__main__':
    # genres = ['family', 'sci-fi']
    # genres = ['horror', 'sci-fi']
    genres = ['family', 'horror']
    # genres = ['family', 'horror', 'sci-fi']
    df = load_data()
    df = filter_genres(df, genres)

    train_accuracy_scores, test_accuracy_scores = custom_cross_validate(df, k=5)
    print 'train_accuracy_scores', train_accuracy_scores
    print 'test_accuracy_scores', test_accuracy_scores
    print np.mean(train_accuracy_scores)






    # single split

    # df_train, df_test= custom_train_test_split(df, 0.2)

    # y_train, movies_train, X_train = destructure_df(df_train)
    # y_test, movies_test, X_test = destructure_df(df_test)

    # clf = RandomForestClassifier()
    # clf.fit(X_train, y_train)


    # ensemble_prediction = ensemble_predict(clf, X_test, movies_test)
    # ensemble_test = get_movie_genre(y_test, movies_test)
    # print('accuracy:', custom_accuracy_score(ensemble_test, ensemble_prediction))
















    # # shuffle rows for cross-validation
    # df = df.sample(frac=1)

    # y = df.pop('genre').values
    # del df['movie']
    # X = df.values

    # # np.stack([a,b,c], axis=1)

    # # split into train and test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # models = [RandomForestClassifier, GaussianNB]

    # for Model in models:
    #     run_model(Model, X_train, X_test, y_train, y_test)
    #     cross_validate(Model, X, y, cv=5)













