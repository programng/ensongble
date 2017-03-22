from __future__ import print_function

import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
os.environ['LIBROSA_CACHE_LEVEL'] = '50'
import time
import math
import glob
import cPickle
import ipdb

import numpy as np
import pandas as pd
import librosa


# import matplotlib.pyplot as plt
# import matplotlib.style as ms
# ms.use('seaborn-muted')

# IPython.display for audio output
# import IPython.display

# librosa for audio
# display module for visualization
# import librosa.display


########################
### RANDOM FOREST ###
#######################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# path to audio file
# audio_path = "/data/music/batch2/Moana (Original Motion Picture Soundtrack|Deluxe Edition)/04 How Far I'll Go.wav"
# audio_path = "/data/music/batch2/Moana (Original Motion Picture Soundtrack|Deluxe Edition)/04 How Far I'll Go.wav"

# load the song
# offset-- start reading after this time (in seconds)
# duration-- load up to this much audio (in seconds)
# y is audio_buffer
# sr is sampling_rate
# y, sr = librosa.load(audio_path, offset=49.0, duration=25.0)

# play the song
# IPython.display.Audio(data=y, rate=sr)





def time_elapsed(start_time):
    elapsed_time = float(time.time()) - float(start_time)
    minutes = math.floor(elapsed_time / 60)
    seconds = elapsed_time % 60
    print("{} minutes and {} seconds".format(minutes, seconds))

def load_data(genres_list, music_path, sample_number):
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
        temp_count = 1
        genres_dict[genre] = {}
        genre_path = os.path.join(music_path, genre, '*')
        print('loading songs for:', genre, '...')
        audio_buffer_list = []
        for moviename in glob.glob(genre_path):
            movie_path = os.path.join(music_path, genre, moviename, '*')
            print('loading songs for:', moviename, '...')
            for filename in glob.glob(movie_path):
                audio_path = os.path.join(music_path, genre, moviename, filename)
                print('loading song:', filename, '...')
                audio_buffer, sampling_rate = librosa.load(audio_path, offset=10.0, duration=10.0)
                # audio_buffer, sampling_rate = librosa.load(audio_path, offset=20.0, duration=1.0)
                audio_buffer_list.append(audio_buffer)
                print('... finished loading song:', filename)
                temp_count += 1
                if temp_count >= sample_number:
                    break
            print('... finished loading songs for:', moviename)
            if temp_count >= sample_number:
                break
        genres_dict[genre] = [np.array(audio_buffer_list)]
        genres_dict[genre].append(np.array([genre] * len(audio_buffer_list)))
        print('... finished loading songs for:', genre)

    return genres_dict


if __name__ == '__main__':
    start_time = time.time()

    # user provided list of genres they want to load
    genres_list = ['family', 'sci-fi']

    # load dictionary whose keys are genres and values are a list, 0th index is array of songs, 1st index is array of labels
    print('loading audio files...')

    pickle_filename = 'data.pkl'
    pickle_path = os.path.join('/data/music', pickle_filename)
    # pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pickle_filename)
    if (os.path.isfile(pickle_path)):
        print('loading pickled data...')
        f_r = open(pickle_path)
        genres_dict = cPickle.load(f_r)
        f_r.close()
        print('finished loading pickled data')
    else:
        print('loading fresh data...')
        genres_dict = load_data(genres_list, '/data/music', 100)
        time_elapsed(start_time)
        # f_w = open(pickle_path, 'w')
        # cPickle.dump(genres_dict, f_w)
        # f_w.close()
        print('finished loading fresh data')

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

    df = pd.DataFrame(data={'genre': y})

    print('calculating spectral centroids...')
    spectral_centroids = [librosa.feature.spectral_centroid(x) for x in X]
    df['spectral_centroids_mean'] = [spectral_centroid.mean() for spectral_centroid in spectral_centroids]
    df['spectral_centroids_std'] = [spectral_centroid.std() for spectral_centroid in spectral_centroids]
    print('...finished calculating spectral centroids')

    print('calculating mfccs...')
    mfccs = [librosa.feature.mfcc(x, n_mfcc=5) for x in X]
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

    y = df.pop('genre').values
    X = df.values
    # ipdb.set_trace()

    # # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # # instantiate model
    rf = RandomForestClassifier()
    print('fitting random forest...')
    rf.fit(X_train, y_train)
    print('...finished fitting random forest')

    # # get accuracy score
    print("accuracy score:", rf.score(X_test, y_test))

    # # get y predictions from test set
    # y_predict = rf.predict(X_test)

    # # get confusion matrix
    # print("confusion matrix:")
    # print(confusion_matrix(y_test, y_predict))

    # # get precision
    # print("precision:", precision_score(y_test, y_predict))

    # # get recall
    # print("recall:", recall_score(y_test, y_predict))











# Silla's feature vector

# TIMBRAL FEATURES


# RHYTHMIC FEATURES

# PITCH FEATURES






