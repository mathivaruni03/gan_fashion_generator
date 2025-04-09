import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_and_save_images

NOISE_DIM = 100
NUM_SAMPLES = 16

# Load your trained generator model
generator = tf.keras.models.load_model('your_saved_generator_model.h5')

# Generate noise and produce images
seed = tf.random.normal([NUM_SAMPLES, NOISE_DIM])
generate_and_save_images(generator, epoch=999, test_input=seed, output_dir='sample_outputs')

print("✔️ Generated samples saved in 'sample_outputs/'")
generator.save('your_saved_generator_model.h5')
