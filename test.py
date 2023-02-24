from autoencoder import get_model
from utils import create_dataset, generator, savitzky_golay
from config import *
import os
import matplotlib.pyplot as plt
import numpy as np


filenames = []
filenames += [os.path.join(dataset_path, file_name) for file_name in os.listdir(dataset_path) if (file_name.endswith('tfrecords'))]

dataset = create_dataset(filenames)
it = iter(dataset)

model = get_model()

for batch in generator(it):
    output = model(batch[0])
    break

output = np.squeeze(output[0])
inputs = np.squeeze(batch[0].numpy())

plt.rcParams["figure.figsize"] = (7,3)
plt.plot(output[0][1512:1868], 'red', label = 'output')
plt.plot(inputs[0][1512:1868], 'blue', label = 'input')
plt.savefig('testsg.pdf')
