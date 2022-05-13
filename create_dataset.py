import numpy as np
import os
import librosa
import tensorflow as tf

split_name = "Train_valid_split.txt"

root = "VS10"
dataset_path = "VS10_tf"
SR = 44100
SIGNAL_LENGTH = 10
SAMPLES = 442368
TO_PAD = SAMPLES - SR * SIGNAL_LENGTH

def make_example(signal, speed, instant):
    features_dict = {"signal": tf.train.Feature(float_list=tf.train.FloatList(value=signal)), "speed": tf.train.Feature(float_list=tf.train.FloatList(value=[speed])), "instant": tf.train.Feature(float_list=tf.train.FloatList(value=[instant]))}
    features = tf.train.Features(feature=features_dict)
    return tf.train.Example(features=features).SerializeToString()

def load_audio(audio_filename, signal_length=None, sample_rate=44100):
    y, fs = librosa.load(audio_filename, sr=None)
    if y.dtype == np.int16:
        y = y / 32768.0  # short int to float
    if len(y.shape) == 2:
        y = y[:, 0]
    y = np.asarray(y)

    if signal_length is not None:
        if y.size / fs > signal_length:
            # Cut the signal if it's too long
            y = y[: int(signal_length * fs)]
        else:
            # Pad the signal with zeros if it's too short
            y = np.pad(y, int((signal_length * fs - y.size) / 2), mode='constant')

    if fs != sample_rate:
        y = librosa.core.resample(y, fs, sample_rate)
        fs = sample_rate

    return y, fs

def create_dataset():
    dirs = os.listdir(root)
    name = dirs[0]
    tfRecord_train_filename = os.path.join(dataset_path, name + "_train" + ".tfrecords")
    tfRecord_valid_filename = os.path.join(dataset_path, name + "_valid" + ".tfrecords")
    writer_train = tf.io.TFRecordWriter(tfRecord_train_filename)
    writer_valid = tf.io.TFRecordWriter(tfRecord_valid_filename)
    for dir in dirs:
        if name != dir:
            name = dir
            tfRecord_train_filename = os.path.join(dataset_path, name + "_train" + ".tfrecords")
            tfRecord_valid_filename = os.path.join(dataset_path, name + "_valid" + ".tfrecords")
            writer_train = tf.io.TFRecordWriter(tfRecord_train_filename)
            writer_valid = tf.io.TFRecordWriter(tfRecord_valid_filename)
        dir_path = os.path.join(root, dir)
        split_path = os.path.join(dir_path, split_name)
        split = open(split_path, 'r')
        split_lines = split.readlines()
        for split_line in split_lines:
            data = split_line.split()
            audio_filename = data[0] + ".wav"
            text_filename = data[0] + ".txt"
            status = data[1]
            audio_path = os.path.join(dir_path, audio_filename)
            text_path = os.path.join(dir_path, text_filename)
            y, _ = load_audio(audio_path, SIGNAL_LENGTH)
            y = np.pad(y, (0, TO_PAD))
            text = open(text_path, 'r')
            lines = text.readlines()
            text_data = lines[0].split()
            speed = float(text_data[0])
            instant = float(text_data[1])
            if status == "train":
                writer_train.write(make_example(y, speed, instant))
            else:
                writer_valid.write(make_example(y, speed, instant))

def main():
    create_dataset()

if __name__ == '__main__':
    main()
