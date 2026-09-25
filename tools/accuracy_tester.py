import numpy as np
import barebones_ml as bbml
from utils import load_mnist_csv
import argparse

parser = argparse.ArgumentParser(
        description="Test the accuracy of an MNIST prediction model"
    )

parser.add_index = parser.add_argument(
        "model_path", 
        type=str, 
        help="The path to the chosen model folder (e.g., ../models/v1)."
    )

args = parser.parse_args()
model_path = args.model_path

network = bbml.Network(bbml.Losses.softmax_cross_entropy)
network.add(bbml.PreTrainedLayer(f"{model_path}/layer_0_weights.npy", f"{model_path}/layer_0_biases.npy", bbml.Activations.relu))
network.add(bbml.PreTrainedLayer(f"{model_path}/layer_1_weights.npy", f"{model_path}/layer_1_biases.npy", bbml.Activations.relu))
network.add(bbml.PreTrainedLayer(f"{model_path}/layer_2_weights.npy", f"{model_path}/layer_2_biases.npy", bbml.Activations.linear))

images, labels = load_mnist_csv("../data/mnist_test.csv")

predictions = network.forward(images)
guessed_numbers = np.argmax(predictions, axis=1)
actual_numbers = np.argmax(labels, axis=1)

accuracy = np.mean(guessed_numbers == actual_numbers) * 100
print(f"accuracy: {accuracy:.2f}%")