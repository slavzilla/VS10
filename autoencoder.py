import tensorflow as tf
import numpy as np
import os
from random import shuffle
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Conv1DTranspose, Concatenate, AlphaDropout
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import LecunNormal
from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh
from tensorflow.keras.optimizers import Nadam
import matplotlib.pyplot as plt
from utils import _parse_function_, generator
from config import *


def compute_loss(model, input):
  ae, _ = model(input)
  return MeanAbsoluteError()(input, ae)

def train_step(model, input, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, input)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss


def double_conv_block(signal, kernel, filters):
  output = Conv1D(filters=filters, kernel_size=kernel, padding='same', activation='selu', kernel_initializer=LecunNormal())(signal)
  output = Conv1D(filters=filters, kernel_size=kernel, padding='same', activation='selu', kernel_initializer=LecunNormal())(output)
  return output

def downsample_block(signal, kernel, filters, downsample):
  f = double_conv_block(signal, kernel, filters)
  p = MaxPool1D(downsample)(f)
  p = AlphaDropout(rate=0.3)(p)
  return f, p

def upsample_block(signal, features, kernel, filters, upsample):
  output = Conv1DTranspose(filters=filters, kernel_size=kernel, strides=upsample, padding='same', activation='selu', kernel_initializer=LecunNormal())(signal)
  output = Concatenate()([output, features])
  output = AlphaDropout(rate=0.3)(output)
  output = double_conv_block(output, kernel, filters)
  return output

def get_ae(ss=[SIGNAL_LENGTH, 1]):
  input=Input(shape=ss)
  f1, p1 = downsample_block(input, 441, 8, 32)
  f2, p2 = downsample_block(p1, 61, 16, 8)
  f3, p3 = downsample_block(p2, 19, 32, 2)
  f4, p4 = downsample_block(p3, 11, 64, 2)
  f5, p5 = downsample_block(p4, 9, 128, 2)
  f6, p6 = downsample_block(p5, 3, 256, 2)
  bottleneck = double_conv_block(p6, 3, 512)
  u8 = upsample_block(bottleneck, f6, 3, 256, 2)
  u9 = upsample_block(u8, f5, 9, 128, 2)
  u10 = upsample_block(u9, f4, 11, 64, 2)
  u11 = upsample_block(u10, f3, 19, 32, 2)
  u12 = upsample_block(u11, f2, 61, 16, 8)
  u13 = upsample_block(u12, f1, 441, 8, 32)

  output = Conv1D(filters=1, kernel_size=441, strides=1, padding='same', activation=tanh, kernel_initializer=GLOROT_INITIALIZER)(u13)

  return  Model(inputs = input, outputs = [output, bottleneck])

def train():
    filenames = []
    filenames += [os.path.join(dataset_path, file_name) for file_name in os.listdir(dataset_path) if (file_name.endswith('tfrecords'))]

    train_dataset = list(filter(lambda filename: "train" in filename, filenames))

    shuffle(filenames)
    dataset = tf.data.TFRecordDataset(train_dataset)
    dataset = dataset.map(_parse_function_)
    dataset = dataset.repeat(count=80)
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    it = iter(dataset)

    autoencoder = get_ae()
    input = Input((SIGNAL_LENGTH, 1))
    ae = autoencoder(input)
    model = Model(inputs=input, outputs=ae)
    optimizer = Nadam(learning_rate=1*1e-5)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, it=it)
    manager = tf.train.CheckpointManager(ckpt, "vs10_ae", max_to_keep=2)
    manager.restore_or_initialize()

    step = 0
    losses = []
    for batch in generator(it):
       loss = train_step(model, batch[0], optimizer)
       losses.append(loss.numpy())
       step += 1
       print("loss is: ", losses[-1], "batch number is:", step)
       if (step % EPOCH == 0):
           manager.save()
    np.save("losses.npy", losses)

def get_model():
  autoencoder = get_ae()
  input = Input((SIGNAL_LENGTH, 1))
  ae = autoencoder(input)
  model = Model(inputs=input, outputs=ae)
  model.trainable = False
  ckpt = tf.train.Checkpoint(model=model)
  manager = tf.train.CheckpointManager(ckpt, "vs10_ae", max_to_keep=2)
  manager.restore_or_initialize()
  return  model


def main():
    train()

if __name__ == '__main__':
    main()
