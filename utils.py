import tensorflow as tf
from random import shuffle
from config import SIGNAL_LENGTH, NUM_EPOCHS, BUFFER_SIZE, BATCH_SIZE

def _parse_function_(example):
    features_description = {"signal": tf.io.FixedLenFeature(SIGNAL_LENGTH, tf.float32), "speed": tf.io.FixedLenFeature(1, tf.float32), "instant": tf.io.FixedLenFeature(1, tf.float32)}

    features_dict = tf.io.parse_single_example(example, features_description)

    return (tf.expand_dims(features_dict["signal"], axis=-1), features_dict["speed"], features_dict["instant"])

def generator(iterator):
  try:
    while True:
      yield next(iterator)
  except (RuntimeError, StopIteration):
    return

def create_dataset(filenames, num_epochs=NUM_EPOCHS, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, to_shuffle=True):
  if(to_shuffle):
    shuffle(filenames)
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parse_function_)
  dataset = dataset.repeat(count=num_epochs)
  if(to_shuffle):
    dataset = dataset.shuffle(buffer_size=buffer_size)
  dataset = dataset.batch(batch_size, drop_remainder=False)
  return dataset

def range_index(value, ranges):
  ind = 0
  for limit in ranges:
    if value < limit:
      return ind
    
    ind+=1
  
  return ind

def test_validation_split(data, speed_ranges=[37.5,  45.0,  52.5,  60.0,  67.5,  75.0,  82.5,  90.0, 97.5]):
  val_sound_data = []
  val_speed_data = []
  test_sound_data = []
  test_speed_data = []
  
  checked_ranges = [False]*(len(speed_ranges)+1)

  for i in range(data[1].shape[0]):
    speed = data[1][i].numpy()
    range_ind = range_index(speed, speed_ranges)
    if checked_ranges[range_ind] is False:
      val_sound_data.append(data[0][i])
      val_speed_data.append(data[1][i])
      checked_ranges[range_ind] = True
    else:
      test_sound_data.append(data[0][i])
      test_speed_data.append(data[1][i])
      

  val_sound_data = tf.stack(val_sound_data, axis=0)
  val_speed_data = tf.stack(val_speed_data, axis=0)
  test_sound_data = tf.stack(test_sound_data, axis=0)
  test_speed_data = tf.stack(test_speed_data, axis=0)

  return test_sound_data, test_speed_data, val_sound_data, val_speed_data

def normalize(input, mu, sigma):
  return (input - mu) / sigma

def denormalize(input, mu, sigma):
  return input * sigma + mu