SIGNAL_LENGTH = 442368
AE_O_SHAPE = (108, 512, 1)
dataset_path = "VS10_tf"
BATCH_SIZE = 32
BATCH_COEFF = 4
EPOCH_LEN = 304
EPOCH = int(EPOCH_LEN / BATCH_SIZE)
BUFFER_SIZE = 64
NUM_EPOCHS = 40
NUM_EPOCHS = 120
ES_STEP = 1
PATIENCE = NUM_EPOCHS
GLOROT_INITIALIZER = 'glorot_normal'

cars = ['VWPassat', 'RenaultScenic', 'RenaultCaptur', 'Peugeot3008', 'Peugeot307', 'OpelInsignia', 'NissanQashqai', 'MercedesAMG550', 'Mazda3', 'CitroenC4Picasso']
lengths = {
  'VWPassat': 35,
  'RenaultScenic': 35,
  'RenaultCaptur': 33,
  'Peugeot3008': 31,
  'Peugeot307': 29,
  'OpelInsignia': 27,
  'NissanQashqai': 29,
  'MercedesAMG550': 30,
  'Mazda3': 32,
  'CitroenC4Picasso': 23
}