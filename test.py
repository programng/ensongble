import os

if __name__ == '__main__':
    print os.getcwd()
    print os.path.basename(__file__)
    print os.path.abspath(__file__)
    print os.path.dirname(__file__)
    print os.path.dirname(os.path.abspath(__file__))



import glob
import os
import librosa
from multiprocessing import Pool
family_path = genre_path = os.path.join('/data/music/', 'family', '*')

pool = Pool(processes=16)

files = []

for moviename in glob.glob(family_path):
    for songname in glob.glob(moviename + '/*'):
        files.append(songname)

res = pool.map(librosa.load, files[:10])

%time pool.map(librosa.load,files[:10])
CPU times: user 84 ms, sys: 60 ms, total: 144 ms
Wall time: 13.7 s









def test():
     for f in files[:10]:
         librosa.load(f)

%time test()

CPU times: user 57.6 s, sys: 96 ms, total: 57.7 s
Wall time: 57.7 s
