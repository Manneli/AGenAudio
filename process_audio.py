import os.path

import librosa.feature
import numpy
import numpy as np
import soundfile

DATA_PATH = "/home/anneli/AGenAudio/wav_traindevel"
PROCESSED_PATH = "/home/anneli/AGenAudio/"
DATASET = "/home/anneli/AGenAudio/trainSampleList_devel_forUse.txt"

win = 0.025
hop = 0.01
feat = 13


def main():
    produce_from_raw(DATA_PATH, new_dir=PROCESSED_PATH)

    train = []
    test = []
    val = []

    pers = []

    data = open(DATASET, "r")
    line = data.readline()
    while line is not "":
        spline = line.split(" ")
        if [int(spline[3]),spline[4] [:-1],int(spline[2])] not in pers :
            pers = pers + [ [int(spline[3]),spline[4] [:-1],int(spline[2])] ]
        line = data.readline()

    pers.sort(key=lambda x: x[0])

    for i in range (299):
        if i % 5 == 3:
            val.append(pers.pop(0))
        elif i % 5 == 4:
            test.append(pers.pop(0))
        else:
            train.append(pers.pop(0))

    data = open(os.path.join(PROCESSED_PATH,"train.dataset"), "w")
    for trainable in train:
        for word in os.listdir(os.path.join(PROCESSED_PATH,"wav_traindevel_processed",str(trainable[2]))):
            for wave in os.listdir(os.path.join(PROCESSED_PATH,"wav_traindevel_processed",str(trainable[2]),word)):
                if (wave[-4:] == ".npy" and wave[-8:] != "Norm.npy"):
                    data.write( os.path.join(PROCESSED_PATH,"wav_traindevel_processed",str(trainable[2]),word,wave) + " {} {} {}\n".format(trainable[2], trainable[0],trainable[1]))

    data = open(os.path.join(PROCESSED_PATH, "test.dataset"), "w")
    for trainable in test:
        for word in os.listdir(os.path.join(PROCESSED_PATH, "wav_traindevel_processed", str(trainable[2]))):
            for wave in os.listdir(os.path.join(PROCESSED_PATH, "wav_traindevel_processed", str(trainable[2]), word)):
                if (wave[-4:] == ".npy" and wave[-8:] != "Norm.npy"):
                    data.write(os.path.join(PROCESSED_PATH, "wav_traindevel_processed", str(trainable[2]), word,
                                            wave) + " {} {} {}\n".format(trainable[2], trainable[0], trainable[1]))

    data = open(os.path.join(PROCESSED_PATH, "validation.dataset"), "w")
    for trainable in val:
        for word in os.listdir(os.path.join(PROCESSED_PATH, "wav_traindevel_processed", str(trainable[2]))):
            for wave in os.listdir(os.path.join(PROCESSED_PATH, "wav_traindevel_processed", str(trainable[2]), word)):
                if (wave[-4:] == ".npy" and wave[-8:] != "Norm.npy"):
                    data.write(os.path.join(PROCESSED_PATH, "wav_traindevel_processed", str(trainable[2]), word,
                                            wave) + " {} {} {}\n".format(trainable[2], trainable[0], trainable[1]))




# Searches all directories under Path and saves .wav,.npy for every .raw in it
# If wav is True, Wave data is saved; if mcff is True the not normalized spectrum is saved; if normMfcc is True, the
# normalized Spectrum is saved
# if new_dir is not "" there will be a new directory copying the old structure at root new_dir
# else it will be saved next to the raw data
# new dir must be "" or preexist
# if norm_axis is True, the normalisation is only done for the features (axis = 1)
# TODO relative Pfade
def produce_from_raw(path, channels=1, sr=8000, subtype="PCM_16", new_dir="", wav=True, mfcc=False, normMfcc=True,
                norm_ax=False):
    if not os.path.exists(path):
        raise AttributeError("Need existing path")

    if new_dir != "":
        if not os.path.exists(new_dir):
            raise AttributeError("Need existing path")

    if os.path.isdir(path):
        if new_dir != "":
            name = os.path.split(path)[1]
            new_dir = os.path.join(new_dir, name + "_processed")
            make_dir(new_dir)
        produce_from_raw_intern(path, channels=channels, sr=sr, subtype=subtype, new_dir=new_dir, wav=wav, mfcc=mfcc,
                           normMfcc=normMfcc, norm_ax=norm_ax) 
    elif path[-4:] == ".raw":
        name = os.path.split(path)[1]
        wave, sr = soundfile.read(path, channels=channels, samplerate=sr, subtype=subtype)
        newdir = path if new_dir == "" else os.path.join(new_dir, name)
        if wav:
            soundfile.write(os.path.join(newdir[:-4] + ".wav"), wave, sr, subtype=subtype)
        win_length = int(sr * win)
        hop_length = int(sr * hop)
        melfreq = librosa.feature.mfcc(y=wave, sr=sr, hop_length=hop_length, win_length=win_length, n_mfcc=feat)
        if mfcc:
            numpy.save(newdir[:-4] + "_notNorm.npy", melfreq)
        if norm_ax:
            melfreq = (melfreq - melfreq.mean(axis=1, keepdims=True)) / melfreq.std(axis=1, keepdims=True)
        else:
            melfreq = (melfreq - melfreq.mean()) / melfreq.std()
        if normMfcc:
            numpy.save(newdir[:-4] + ".npy", melfreq)


def produce_from_raw_intern(path, channels=1, sr=8000, subtype="PCM_16", new_dir="", wav=True, mfcc=False, normMfcc=True,
                       norm_ax=False):
    for data in os.listdir(path):
        if os.path.isdir(os.path.join(path, data)):

            if new_dir != "":
                newdir = os.path.join(new_dir, data)
                make_dir(newdir)
            else:
                newdir = new_dir

            produce_from_raw_intern(os.path.join(path, data), channels=channels, sr=sr, subtype=subtype,
                               new_dir=newdir, wav=wav, mfcc=mfcc, normMfcc=normMfcc, norm_ax=norm_ax)
        elif data[-4:] == ".raw":
            wave, sr = soundfile.read(os.path.join(path, data), channels=channels, samplerate=sr, subtype=subtype)
            newdir = path if new_dir == "" else new_dir
            if wav:
                soundfile.write(os.path.join(newdir, data[:-4] + ".wav"), wave, sr, subtype=subtype)
            win_length = int(sr * win)
            hop_length = int(sr * hop)
            melfreq = librosa.feature.mfcc(y=wave, sr=sr, hop_length=hop_length, win_length=win_length, n_mfcc=feat)
            if mfcc:
                numpy.save(os.path.join(newdir, data[:-4] + "_notNorm.npy"), melfreq)
            if norm_ax:
                melfreq = (melfreq - melfreq.mean(axis=1, keepdims=True)) / melfreq.std(axis=1, keepdims=True)
            else:
                melfreq = (melfreq - melfreq.mean()) / melfreq.std()
            if normMfcc:
                numpy.save(os.path.join(newdir, data[:-4] + ".npy"), melfreq)



def make_dir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print("made directory: ", path)
        os.mkdir(path)


if __name__ == "__main__":
    main()
