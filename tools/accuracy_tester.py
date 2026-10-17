import numpy as np
import barebones_ml as bbml
from utils import load_mnist_csv, softmax
import argparse
import os
from pathlib import Path
import json

parser = argparse.ArgumentParser(
        description="Test the accuracy of an MNIST prediction model"
    )

parser.add_argument(
        "model_path", 
        type=str, 
        help="The path to the chosen model folder (e.g., ../models/v1)."
    )

args = parser.parse_args()
model_path = Path(args.model_path)

amount_layers = 0
activations = []
with open(model_path / "model.json") as f:
    raw_unloaded_data = json.load(f)
    amount_layers = raw_unloaded_data["amount_layers"]
    activations = raw_unloaded_data["activations"]

network = bbml.Network("softmax_cross_entropy")
for i in range(amount_layers):
    network.add(bbml.PreTrainedLayer(model_path / f"layer_{i}_weights.npy", model_path / f"layer_{i}_biases.npy", activations[i]))

images, labels = load_mnist_csv("../data/mnist_test.csv")

predictions = network.forward(images)
softmaxed_predictions = softmax(predictions)

guessed_numbers = np.argmax(predictions, axis=1)
actual_numbers = np.argmax(labels, axis=1)
accuracy = np.mean(guessed_numbers == actual_numbers) * 100
print(f"accuracy: {accuracy:.2f}%")

wrong_indexes = np.where(guessed_numbers != actual_numbers)[0]
fails = []

for idx in wrong_indexes:
    predicted_class = int(guessed_numbers[idx])
    confidence = float(softmaxed_predictions[idx][predicted_class])
    fail_entry = {
        "image_id": int(idx),
        "actual": int(actual_numbers[idx]),
        "guessed": predicted_class,
        "confidence": round(confidence * 100, 4)
    }
    fails.append(fail_entry)

hall_of_fails = {}
hall_of_fails["model"] = model_path.name
hall_of_fails["fails"] = fails

os.makedirs("../outputs", exist_ok=True)
with open("../outputs/fails.json", "w") as f:
    json.dump(hall_of_fails, f)