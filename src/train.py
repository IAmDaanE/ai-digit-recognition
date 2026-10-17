import numpy as np
import barebones_ml as bbml
from utils import load_mnist_csv, shuffle_dataset
import argparse
from pathlib import Path
import json
import pygame

parser = argparse.ArgumentParser(
        description="Train an MNIST prediction model."
    )

parser.add_index = parser.add_argument(
        "model_path", 
        type=str, 
        help="The path to the folder where the resulting trained model will be saved (e.g., ../models/v1)."
    )

args = parser.parse_args()
model_path = Path(args.model_path)
model_path.mkdir(parents=True, exist_ok=True)

pygame.init()
screen = pygame.display.set_mode((980, 580))
pygame.display.set_caption("MNIST Model Training")

network = bbml.Network("softmax_cross_entropy", model_name=model_path.name)
network.add(bbml.Layer(784, 256, "relu", "he"))
network.add(bbml.Layer(256, 256, "relu", "he"))
network.add(bbml.Layer(256, 256, "relu", "he"))
network.add(bbml.Layer(256, 10, "linear", "xavier"))

vis = bbml.Visualization()

images, labels = load_mnist_csv("../data/mnist_train.csv")

batch_size = 64

def run_training(epochs):
    global images, labels
    current_lr = 0.0005
    try:
        for epoch in range(epochs):
            network.epoch = epoch
            current_lr = bbml.LRDecays.exponential_decay(current_lr, 0.96)
            network.current_lr = current_lr
            images, labels = shuffle_dataset(images, labels)
            for i in range(0, images.shape[0], batch_size):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                images_batch = images[i : i + batch_size]
                labels_batch = labels[i : i + batch_size]
                prediction = network.forward(images_batch)
                loss = network.loss_function(prediction, labels_batch)
                network.loss = loss
                network.backward(prediction, labels_batch)
                network.update(current_lr)
            vis.full_draw(network_class=network, screen=screen, display_legend=True, plot_lr=False, display_axes=False)
            pygame.display.update()
    except KeyboardInterrupt:
        amount_layers = len(network.layers)
        activations = []
        for layer in network.layers:
            activations.append(layer.activation_string)
        with open(model_path / "model.json", "w") as f:
            json.dump({"amount_layers": amount_layers, "activations": activations}, f)
        for i, layer in enumerate(network.layers):
            np.save(model_path / f"layer_{i}_weights.npy", layer.weights)
            np.save(model_path / f"layer_{i}_biases.npy", layer.biases)
        return

if __name__ == "__main__":
    run_training(30000)