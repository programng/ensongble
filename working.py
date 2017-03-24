from __future__ import print_function

import os
# os.environ['LIBROSA_CACHE_DIR'] = '/data/tmp/librosa_cache'
# os.environ['LIBROSA_CACHE_LEVEL'] = '50'
import time
import math
import glob
from multiprocessing import Pool
import ipdb
from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

pool = Pool(processes=16)

def current_time():
    print("current time: {}".format(str(datetime.now().time())))


def time_elapsed(start_time):
    elapsed_time = float(time.time()) - float(start_time)
    minutes = math.floor(elapsed_time / 60)
    seconds = elapsed_time % 60
    current_time()
    print("{} minutes and {} seconds".format(minutes, seconds))

def load_data(genres_list, music_path):
    time_elapsed(start_time)
    """Load audio buffer (data) for all songs for the specified genres

    Keyword arguments:
    genres_list -- list of genre names(str)
    music_path -- the path to the music(str)

    Notes:
    The file structure follows this form:
    music_path
    ----genre
    --------song

    Dictionary returned looks like:
    {
        'genre1': [[], []], # first list is X, second is y
        'genre2': [[], []] # first list is X, second is y
    }
    """
    # dictionary whose keys are genres and values are  a tuple, 0th index is array of songs, 1st index is array of labels
    genres_dict = {}
    # add the data for each genre to the dictionary
    for genre in genres_list:

        print('loading songs for:', genre, '...')

        pickle_filename = '{}.npy'.format(genre)
        pickle_path = os.path.join(music_path, pickle_filename)
        if (os.path.isfile(pickle_path)):
            print('loading pickled {} data...'.format(genre))
            time_elapsed(start_time)
            audio_buffer_array_for_genre = np.load(pickle_path)
            print('finished loading pickled {} data'.format(genre))
            time_elapsed(start_time)
        else:
            print('loading fresh data...')
            all_songs_for_genre = []
            genre_path = os.path.join(music_path, genre, '*')

            for moviename in glob.glob(genre_path):
                movie_path = os.path.join(music_path, genre, moviename, '*')
                all_songs_for_movie = glob.glob(movie_path)
                all_songs_for_genre += all_songs_for_movie

            loaded_audio_for_genre = pool.map(librosa.load, all_songs_for_genre)
            audio_buffer_array_for_genre = np.array([loaded_audio[0] for loaded_audio in loaded_audio_for_genre])
            print('pickling {} data...'.format(genre))
            np.save(pickle_path, audio_buffer_array_for_genre)
            print('...finished pickling {} data'.format(genre))
            print('finished loading fresh data')

        genres_dict[genre] = [audio_buffer_array_for_genre]
        genres_dict[genre].append(np.array([genre] * len(audio_buffer_array_for_genre)))
        print('... finished loading songs for:', genre)

    return genres_dict

def run_model(Model, X_train, X_test, y_train, y_test):
    name = Model.__name__
    print('##################')
    print('##################')
    print('fitting {}...'.format(name))
    clf = Model()
    clf.fit(X_train, y_train)
    print('...finished fitting {}'.format(name))
    y_predict = clf.predict(X_test)
    print(1, "accuracy score:", clf.score(X_test, y_test))

    print("confusion matrix:")
    print(confusion_matrix(y_test, y_predict))

    le = LabelEncoder()
    le.fit(y_train)
    le_y_test = le.transform(y_test)
    le_y_predict = le.transform(y_predict)
    print("precision:", precision_score(le_y_test, le_y_predict))

    print(2, "accuracy score:", accuracy_score(le_y_test, le_y_predict))

    # get recall
    print("recall:", recall_score(le_y_test, le_y_predict))

    if name == 'RandomForestClassifier':
        print('RandomForestClassifier feature importance', clf.feature_importances_)

def cross_validate(Model, X, y, cv=5):
    name = Model.__name__
    print('starting cross validation for {}, k=5...'.format(name))
    clf = Model()
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
    print('...finished cross validation for {}, k=5'.format(name))
    print('cross validation scores:', scores)
    print('average of cross validation scores:', scores.mean())

if __name__ == '__main__':
    start_time = time.time()
    current_time()


    # user provided list of genres they want to load
    genres_list = ['family', 'horror', 'sci-fi']
    # genres_list = ['family', 'horror']
    # genres_list = ['family', 'sci-fi']
    print('genres of interest:')
    for genre in genres_list:
        print('    -', genre)


    # load dictionary whose keys are genres and values are a list, 0th index is array of songs, 1st index is array of labels
    print('loading audio files...')

    genres_dict = load_data(genres_list, '/data/music/pkl')
    time_elapsed(start_time)

    print('...finished loading audio files')

    # get X and y
    Xs = []
    ys = []
    for genre_data in genres_dict.values():
        Xs.append(genre_data[0])
        ys.append(genre_data[1])

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    print('X', X)
    print('y', y)

###############################
##### CALCULATE FEATURES #####
###############################

    df = pd.DataFrame(data={'genre': y})

    print('calculating spectral rolloff...')
    time_elapsed(start_time)
    pickle_filename = 'spectral_rolloffs_{}.npy'
    pickle_path_mean = os.path.join('/data/music/features_pkl', pickle_filename.format('mean'))
    pickle_path_std = os.path.join('/data/music/features_pkl', pickle_filename.format('std'))
    if (os.path.isfile(pickle_path_mean) and os.path.isfile(pickle_path_std) ):
        print('loading pickled data...')
        df['spectral_rolloffs_mean'] = np.load(pickle_path_mean)
        df['spectral_rolloffs_std'] = np.load(pickle_path_std)
        print('finished loading pickled data')
    else:
        spectral_rolloffs = pool.map(librosa.feature.spectral_rolloff, X)
        df['spectral_rolloffs_mean'] = [spectral_rolloff.mean() for spectral_rolloff in spectral_rolloffs]
        df['spectral_rolloffs_std'] = [spectral_rolloff.std() for spectral_rolloff in spectral_rolloffs]
        np.save(pickle_path_mean, df['spectral_rolloffs_mean'])
        np.save(pickle_path_std, df['spectral_rolloffs_std'])
    # spectral_rolloffs = [librosa.feature.spectral_rolloff(x) for x in X]
    print('...finished calculating spectral rolloff')
    time_elapsed(start_time)

    print('calculating spectral centroids...')
    time_elapsed(start_time)
    pickle_filename = 'spectral_centroids_{}.npy'
    pickle_path_mean = os.path.join('/data/music/features_pkl', pickle_filename.format('mean'))
    pickle_path_std = os.path.join('/data/music/features_pkl', pickle_filename.format('std'))
    if (os.path.isfile(pickle_path_mean) and os.path.isfile(pickle_path_std) ):
        print('loading pickled data...')
        df['spectral_centroids_mean'] = np.load(pickle_path_mean)
        df['spectral_centroids_std'] = np.load(pickle_path_std)
        print('finished loading pickled data')
    else:
        spectral_centroids = pool.map(librosa.feature.spectral_centroid, X)
        df['spectral_centroids_mean'] = [spectral_centroid.mean() for spectral_centroid in spectral_centroids]
        df['spectral_centroids_std'] = [spectral_centroid.std() for spectral_centroid in spectral_centroids]
        np.save(pickle_path_mean, df['spectral_centroids_mean'])
        np.save(pickle_path_std, df['spectral_centroids_std'])
    # spectral_centroids = [librosa.feature.spectral_centroid(x) for x in X]
    print('...finished calculating spectral centroids')
    time_elapsed(start_time)

    print('calculating zero-crossing rate...')
    time_elapsed(start_time)
    pickle_filename = 'zero_crossing_rates_{}.npy'
    pickle_path_mean = os.path.join('/data/music/features_pkl', pickle_filename.format('mean'))
    pickle_path_std = os.path.join('/data/music/features_pkl', pickle_filename.format('std'))
    if (os.path.isfile(pickle_path_mean) and os.path.isfile(pickle_path_std) ):
        print('loading pickled data...')
        df['zero_crossing_rates_mean'] = np.load(pickle_path_mean)
        df['zero_crossing_rates_std'] = np.load(pickle_path_std)
        print('finished loading pickled data')
    else:
        zero_crossing_rates = pool.map(librosa.feature.zero_crossing_rate, X)
        df['zero_crossing_rates_mean'] = [zero_crossing_rate.mean() for zero_crossing_rate in zero_crossing_rates]
        df['zero_crossing_rates_std'] = [zero_crossing_rate.std() for zero_crossing_rate in zero_crossing_rates]
        np.save(pickle_path_mean, df['zero_crossing_rates_mean'])
        np.save(pickle_path_std, df['zero_crossing_rates_std'])
    # zero_crossing_rates = [librosa.feature.zero_crossing_rate(x) for x in X]
    print('...finished calculating zero-crossing rate')
    time_elapsed(start_time)

    print('calculating mfccs...')
    time_elapsed(start_time)

    pickle_filename = 'mfccs_{number}_{metric}.npy'
    pickle_path_mean_1 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=1, metric='mean'))
    pickle_path_std_1 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=1, metric='std'))
    pickle_path_mean_2 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=2, metric='mean'))
    pickle_path_std_2 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=2, metric='std'))
    pickle_path_mean_3 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=3, metric='mean'))
    pickle_path_std_3 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=3, metric='std'))
    pickle_path_mean_4 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=4, metric='mean'))
    pickle_path_std_4 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=4, metric='std'))
    pickle_path_mean_5 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=5, metric='mean'))
    pickle_path_std_5 = os.path.join('/data/music/features_pkl', pickle_filename.format(number=5, metric='std'))

    if (os.path.isfile(pickle_path_mean_1) and os.path.isfile(pickle_path_std_1) ):
        print('loading pickled data...')
        df['mfcc1_mean'] = np.load(pickle_path_mean_1)
        df['mfcc1_std'] = np.load(pickle_path_mean_1)
        df['mfcc2_mean'] = np.load(pickle_path_mean_2)
        df['mfcc2_std'] = np.load(pickle_path_mean_2)
        df['mfcc3_mean'] = np.load(pickle_path_mean_3)
        df['mfcc3_std'] = np.load(pickle_path_mean_3)
        df['mfcc4_mean'] = np.load(pickle_path_mean_4)
        df['mfcc4_std'] = np.load(pickle_path_mean_4)
        df['mfcc5_mean'] = np.load(pickle_path_mean_5)
        df['mfcc5_std'] = np.load(pickle_path_mean_5)
        print('finished loading pickled data')
    else:
        partial_mfcc = partial(librosa.feature.mfcc, n_mfcc=5)
        mfccs = pool.map(partial_mfcc, X)
        # mfccs = [librosa.feature.mfcc(x, n_mfcc=5) for x in X]
        print('...finished calculating mfccs')
        print('calculating mfcc1...')
        mfcc1s = [mfcc[0] for mfcc in mfccs]
        df['mfcc1_mean'] = [mfcc1.mean() for mfcc1 in mfcc1s]
        df['mfcc1_std'] = [mfcc1.std() for mfcc1 in mfcc1s]
        print('...finished calculating mfcc1')
        print('calculating mfcc2...')
        mfcc2s = [mfcc[1] for mfcc in mfccs]
        df['mfcc2_mean'] = [mfcc2.mean() for mfcc2 in mfcc2s]
        df['mfcc2_std'] = [mfcc2.std() for mfcc2 in mfcc2s]
        print('...finished calculating mfcc2')
        print('calculating mfcc3...')
        mfcc3s = [mfcc[2] for mfcc in mfccs]
        df['mfcc3_mean'] = [mfcc3.mean() for mfcc3 in mfcc3s]
        df['mfcc3_std'] = [mfcc3.std() for mfcc3 in mfcc3s]
        print('...finished calculating mfcc3')
        print('calculating mfcc4...')
        mfcc4s = [mfcc[3] for mfcc in mfccs]
        df['mfcc4_mean'] = [mfcc4.mean() for mfcc4 in mfcc4s]
        df['mfcc4_std'] = [mfcc4.std() for mfcc4 in mfcc4s]
        print('...finished calculating mfcc4')
        print('calculating mfcc5...')
        mfcc5s = [mfcc[4] for mfcc in mfccs]
        df['mfcc5_mean'] = [mfcc5.mean() for mfcc5 in mfcc5s]
        df['mfcc5_std'] = [mfcc5.std() for mfcc5 in mfcc5s]
        print('...finished calculating mfcc5')

        np.save(pickle_path_mean_1, df['mfcc1_mean'])
        np.save(pickle_path_std_1, df['mfcc1_std'])
        np.save(pickle_path_mean_2, df['mfcc2_mean'])
        np.save(pickle_path_std_2, df['mfcc2_std'])
        np.save(pickle_path_mean_3, df['mfcc3_mean'])
        np.save(pickle_path_std_3, df['mfcc3_std'])
        np.save(pickle_path_mean_4, df['mfcc4_mean'])
        np.save(pickle_path_std_4, df['mfcc4_std'])
        np.save(pickle_path_mean_5, df['mfcc5_mean'])
        np.save(pickle_path_std_5, df['mfcc5_std'])

    print('...finished calculating mfccs')
    time_elapsed(start_time)



########################################
##### SHUFFLE DATA AND TRAIN | TEST #####
########################################

    # shuffle rows for cross-validation
    df = df.sample(frac=1)

    y = df.pop('genre').values
    X = df.values

    # np.stack([a,b,c], axis=1)

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # # instantiate model
    # rf = RandomForestClassifier()
    # print('fitting random forest...')
    # time_elapsed(start_time)
    # rf.fit(X_train, y_train)
    # print('...finished fitting random forest')
    # time_elapsed(start_time)

    # # get accuracy score
    # print("accuracy score:", rf.score(X_test, y_test))

    # # get y predictions from test set
    # y_predict = rf.predict(X_test)

    # # get confusion matrix
    # print("confusion matrix:")
    # print(confusion_matrix(y_test, y_predict))

    # # get precision
    # le = LabelEncoder()
    # le.fit(y_train)
    # le_y_test = le.transform(y_test)
    # le_y_predict = le.transform(y_predict)
    # print("precision:", precision_score(le_y_test, le_y_predict))

    # # get recall
    # print("recall:", recall_score(le_y_test, le_y_predict))

    # print('starting cross validation, k=5...')
    # scores = cross_val_score(rf, X, y, cv=5, n_jobs=-1)
    # print('...finished cross validation')
    # print('cross validation scores:', scores)
    # print('average of cross validation scores:', scores.mean())
    # time_elapsed(start_time)


    models = [RandomForestClassifier, GaussianNB]

    for Model in models:
        run_model(Model, X_train, X_test, y_train, y_test)
        cross_validate(Model, X, y, cv=5)






# Silla's feature vector

# TIMBRAL FEATURES

# RHYTHMIC FEATURES

# PITCH FEATURES






