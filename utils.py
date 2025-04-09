import matplotlib.pyplot as plt
import numpy as np
import os

def generate_and_save_images(model, epoch, test_input, output_dir='generated_samples'):
    predictions = model(test_input, training=False)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0] * 127.5 + 127.5).numpy().astype(np.uint8), cmap='gray')
        plt.axis('off')

    plt.savefig(f'{output_dir}/image_at_epoch_{epoch:03d}.png')
    plt.close()
