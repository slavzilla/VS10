import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Input, Conv1D, AlphaDropout, Conv2D, Flatten, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import LecunNormal, HeNormal
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Nadam
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
  [print(x,y) for x,y in zip(output.numpy(), label.numpy())]
  se = np.sum(np.square(output.numpy() - label.numpy()))
  return se


def get_regressor_v2(ss=[SIGNAL_LENGTH, 1]):

  input=Input(shape=ss)

  conv_0 = Conv1D(filters=8, kernel_size=121, strides=1, padding='same', kernel_initializer=HeNormal())(input)
  conv_0 = BatchNormalization()(conv_0)
  conv_0 = ReLU()(conv_0)
  conv_0 = Dropout(rate=0.2)(conv_0)

  conv_1 = Conv1D(filters=16, kernel_size=61, strides=16, padding='same', kernel_initializer=HeNormal())(conv_0)
  conv_1 = BatchNormalization()(conv_1)
  conv_1 = ReLU()(conv_1)
  conv_1 = Dropout(rate=0.2)(conv_1) 

  conv_2 = Conv1D(filters=32, kernel_size=61, strides=8, padding='same', kernel_initializer=HeNormal())(conv_1)
  conv_2 = BatchNormalization()(conv_2)
  conv_2 = ReLU()(conv_2)
  conv_2 = Dropout(rate=0.2)(conv_2)  

  conv_3 = Conv1D(filters=64, kernel_size=31, strides=4, padding='same', kernel_initializer=HeNormal())(conv_2)
  conv_3 = BatchNormalization()(conv_3)
  conv_3 = ReLU()(conv_3)
  conv_3 = Dropout(rate=0.2)(conv_3)  

  conv_4 = Conv1D(filters=128, kernel_size=31, strides=4, padding='same', kernel_initializer=HeNormal())(conv_3)
  conv_4 = BatchNormalization()(conv_4)
  conv_4 = ReLU()(conv_4)
  conv_4 = Dropout(rate=0.2)(conv_4)  

  conv_5 = Conv1D(filters=256, kernel_size=15, strides=2, padding='same', kernel_initializer=HeNormal())(conv_4)
  conv_5 = BatchNormalization()(conv_5)
  conv_5 = ReLU()(conv_4)
  conv_5 = Dropout(rate=0.2)(conv_5) 

  flatten = Flatten()(conv_5)

  dense_1 = Dense(units=512, kernel_initializer=HeNormal())(flatten)
  dense_1 = BatchNormalization()(dense_1)
  dense_1 = ReLU()(dense_1)
  dense_1 = Dropout(rate=0.2)(dense_1)

  dense_2 = Dense(units=256, kernel_initializer=HeNormal())(dense_1)
  dense_2 = BatchNormalization()(dense_2)
  dense_2 = ReLU()(dense_2)
  dense_2 = Dropout(rate=0.2)(dense_2)

  dense_3 = Dense(units=128, kernel_initializer=HeNormal())(dense_2)
  dense_3 = BatchNormalization()(dense_3)
  dense_3 = ReLU()(dense_3)
  dense_3 = Dropout(rate=0.2)(dense_3)

  dense_4 = Dense(units=32, activation=relu, kernel_initializer=HeNormal())(dense_3)
  dense_4 = Dropout(rate=0.2)(dense_4)

  dense_5 = Dense(units=1, activation=relu, kernel_initializer=HeNormal())(dense_4)

  return  Model(inputs = input, outputs = dense_5)


def get_regressor(shape=(54, 256, 1)):

  input = Input(shape=shape) 

  conv_1 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(input)

  conv_1 = AlphaDropout(rate=0.4)(conv_1)

  conv_2 = Conv2D(filters=32, kernel_size=(3,5), strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_1)

  conv_2 = AlphaDropout(rate=0.4)(conv_2)

  conv_3 = Conv2D(filters=64, kernel_size=(2,5), strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_2)

  conv_3 = AlphaDropout(rate=0.4)(conv_3)

  conv_4 = Conv2D(filters=64, kernel_size=(2,3), strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_3)

  conv_4 = AlphaDropout(rate=0.4)(conv_4)

  conv_5 = Conv2D(filters=128, kernel_size=2, strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_4)

  conv_5 = AlphaDropout(rate=0.4)(conv_5)

  conv_6 = Conv2D(filters=128, kernel_size=2, strides=2, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_5)

  conv_6 = AlphaDropout(rate=0.4)(conv_6)

  conv_7 = Conv2D(filters=256, kernel_size=(1, 2), strides=(1,2), padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_6)

  conv_7 = AlphaDropout(rate=0.4)(conv_7)

  conv_8 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation='selu', kernel_initializer=LecunNormal())(conv_7)

  conv_8 = AlphaDropout(rate=0.4)(conv_8)

  flatten = Flatten()(conv_8)

  dense_1 = Dense(units=32, activation=relu, kernel_initializer=HeNormal())(flatten)

  dense_1 = Dropout(rate=0.5)(dense_1)

  dense_2 = Dense(units=1, activation=relu, kernel_initializer=HeNormal())(dense_1)

  return Model(inputs = input, outputs = dense_2)


def train():
  r_input = Input((SIGNAL_LENGTH, 1))
  results = {car: [] for car in cars}
  for car in cars:

    EPOCH_LEN = TOTAL_LEN - lengths[car]

    BATCH_SIZE = EPOCH_LEN // BATCH_COEFF

    train_data = list(filter(lambda filename: car not in filename, filenames))
    
    car_data = list(filter(lambda filename: car in filename, filenames))
    car_dataset = create_dataset(car_data, num_epochs=1, to_shuffle=True, batch_size=lengths[car])
    data = car_dataset.get_single_element()

    train_dataset = create_dataset(train_data, batch_size=BATCH_SIZE)
    for i in range(5):

      train_it = iter(train_dataset)
      test_sound_data, test_speed_data, val_sound_data, val_speed_data = test_validation_split(data)

      regressor = get_regressor_v2()
      regr = regressor(r_input)
      regr_model = Model(inputs=r_input, outputs=regr)
      optimizer_regr = Nadam(learning_rate=1e-4, schedule_decay=1e-4)

      ckpt_regr = tf.train.Checkpoint(model=regr_model)
      manager_regr = tf.train.CheckpointManager(ckpt_regr, f'vs10_regr_{car}', max_to_keep=1)

      step = 0
      RMSE_BEST = float('inf')
      loc_patience = 0
      print("==============Train Start==============")
      for batch in generator(train_it):
        if loc_patience >= PATIENCE:
          print("==============EarlyStopping!==============")
          break
        sound, speed, _ = batch
        loss = train_step_regressor(regr_model, sound, speed, optimizer_regr).numpy()
        step += 1
        print("loss is: ", loss, "batch number is:", step)
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