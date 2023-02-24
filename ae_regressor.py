import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Input, Conv1D, AlphaDropout, Conv2D, Flatten, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import LecunNormal, HeNormal
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Nadam
from autoencoder import get_model
from config import *
from utils import create_dataset, test_validation_split, generator

TOTAL_LEN = 0
for car in lengths:
  TOTAL_LEN += lengths[car]

filenames = []
filenames += [os.path.join(dataset_path, file_name) for file_name in os.listdir(dataset_path) if (file_name.endswith('tfrecords'))]


def compute_regressor_loss(model, input, label):
  output = model(input)
  return tf.math.sqrt(MeanSquaredError()(label, output))

def train_step_regressor(model, input, label, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_regressor_loss(model, input, label)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def test_step_regressor(model, input, label):
  output = model(input)
  se = np.sum(np.square(output.numpy() - label.numpy()))
  return se

def get_ae_output(ae, input):
    return np.expand_dims(ae(input)[1].numpy(), -1)

def get_regressor(shape=AE_O_SHAPE):

  input = Input(shape=shape) 

  conv_1 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='selu', kernel_initializer=LecunNormal())(input)

  conv_1 = AlphaDropout(rate=0.4)(conv_1)

  conv_2 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_1)

  conv_2 = AlphaDropout(rate=0.4)(conv_2)

  conv_3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_2)

  conv_3 = AlphaDropout(rate=0.4)(conv_3)

  conv_4 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_3)

  conv_4 = AlphaDropout(rate=0.4)(conv_4)

  conv_5 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_4)

  conv_5 = AlphaDropout(rate=0.4)(conv_5)

  conv_6 = Conv2D(filters=128, kernel_size=2, strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_5)

  conv_6 = AlphaDropout(rate=0.4)(conv_6)

  conv_7 = Conv2D(filters=256, kernel_size=(1, 2), strides=1, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_6)

  conv_7 = AlphaDropout(rate=0.4)(conv_7)

  conv_8 = Conv2D(filters=256, kernel_size=(1, 2), strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_7)

  conv_8 = AlphaDropout(rate=0.4)(conv_8)

  flatten = Flatten()(conv_8)

  dense_1 = Dense(units=512, activation='selu', kernel_initializer=LecunNormal())(flatten)
  dense_1 = AlphaDropout(rate=0.2)(dense_1)

  dense_2 = Dense(units=256, activation='selu', kernel_initializer=LecunNormal())(dense_1)
  dense_2 = AlphaDropout(rate=0.2)(dense_2)

  dense_3 = Dense(units=128, activation='selu', kernel_initializer=LecunNormal())(dense_2)
  dense_3 = AlphaDropout(rate=0.2)(dense_3)

  dense_4 = Dense(units=32, activation=relu, kernel_initializer=HeNormal())(dense_3)
  dense_4 = AlphaDropout(rate=0.2)(dense_4)

  dense_5 = Dense(units=1, activation=relu, kernel_initializer=HeNormal())(dense_4)

  return  Model(inputs = input, outputs = dense_5)


def train():
  r_input = Input(AE_O_SHAPE)
  results = {car: [] for car in cars}
  ae = get_model()
  for car in cars:

    EPOCH_LEN = TOTAL_LEN - lengths[car]

    BATCH_SIZE = EPOCH_LEN // BATCH_COEFF

    train_data = list(filter(lambda filename: car not in filename, filenames))
    
    car_data = list(filter(lambda filename: car in filename, filenames))
    car_dataset = create_dataset(car_data, num_epochs=1, to_shuffle=True, batch_size=lengths[car])
    data = car_dataset.get_single_element()

    train_dataset = create_dataset(train_data, batch_size=BATCH_SIZE)
    for i in range(20):

      train_it = iter(train_dataset)
      test_sound_data, test_speed_data, val_sound_data, val_speed_data = test_validation_split(data)

      test_sound_data = get_ae_output(ae, test_sound_data)
      val_sound_data = get_ae_output(ae, val_sound_data)

      regressor = get_regressor()
      regr = regressor(r_input)
      regr_model = Model(inputs=r_input, outputs=regr)
      optimizer_regr = Nadam(learning_rate=1e-4)

      ckpt_regr = tf.train.Checkpoint(model=regr_model)
      manager_regr = tf.train.CheckpointManager(ckpt_regr, f'vs10_regr_21feb_{car}', max_to_keep=1)

      step = 0
      RMSE_BEST = float('inf')
      loc_patience = 0
      print("==============Train Start==============")
      for batch in generator(train_it):
        sound, speed, _ = batch
        sound = get_ae_output(ae, sound)
        loss = train_step_regressor(regr_model, sound, speed, optimizer_regr).numpy()
        step += 1
        print("loss is: ", loss, "batch number is:", step)
        if (step > 30 and loss > 50):
          print("==============EarlyStopping!==============")
          break
        if (step % BATCH_COEFF == 0 and step % ES_STEP == 0):
          regr_model.trainable = False
          se = test_step_regressor(regr_model, val_sound_data, val_speed_data)
          RMSE = np.sqrt(se / val_sound_data.shape[0])
          print('Validation RMSE for car', car, 'is: ', RMSE, ' while best RMSE is: ', RMSE_BEST)
          if RMSE < RMSE_BEST:
            RMSE_BEST = RMSE
            loc_patience = 0
            manager_regr.save()
          else:
            loc_patience += 1
          regr_model.trainable = True
    
      manager_regr.restore_or_initialize()

      regr_model.trainable = False

      print("==============Testing Start==============")

      se = test_step_regressor(regr_model, test_sound_data, test_speed_data)
      RMSE = np.sqrt(se / test_sound_data.shape[0])
      print('RMSE for car', car, 'is: ', RMSE)
      results[car].append(RMSE)
  print(results)
  np.save("results.npy", results)


def main():
  train()

if __name__ == '__main__':
  main()