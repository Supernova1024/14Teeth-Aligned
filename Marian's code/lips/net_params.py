DATASET_NAME = 'dataset_celeba_4000px'
MODEL_CHOICE = 'deeplab101'

NUM_CLASSES = 2
RE_SIZE = (384, 288)

# hyperparams to tune in order to improve learning
TRAIN_VAL_SPLIT = 0.9
EPOCHS = 50
BATCH_SIZE = 2
LR = 0.0001
GAMMA = 0.95

# transform params to tune in order to artificially increase the dataset
TRANSFORM_PARAMS = {
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1,
    'rotation': 10,
    'affine': 3,
    'perspective': 0.1,
    'p_random_apply': 0.2,
    'p_flip': 0.5,
    'p_noise': 0.01,
}
