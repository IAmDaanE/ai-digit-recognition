import barebones_ml as bbml
import numpy as np
import pygame
import math
import random
from utils import center_drawing, array_to_cords, cords_to_array, softmax
import argparse

parser = argparse.ArgumentParser(
        description="Draw a digit and see what the model thinks it is."
    )

parser.add_index = parser.add_argument(
        "model_path", 
        type=str, 
        help="The path to the folder where the used model will be saved (e.g., ../models/v1)."
)

args = parser.parse_args()
model_path = args.model_path

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 410

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("AI digit recognition")
clock = pygame.time.Clock()

network = bbml.Network(bbml.Losses.softmax_cross_entropy)
network.add(bbml.PreTrainedLayer(f"{model_path}/layer_0_weights.npy", f"{model_path}/layer_0_biases.npy", bbml.Activations.relu))
network.add(bbml.PreTrainedLayer(f"{model_path}/layer_1_weights.npy", f"{model_path}/layer_1_biases.npy", bbml.Activations.relu))
network.add(bbml.PreTrainedLayer(f"{model_path}/layer_2_weights.npy", f"{model_path}/layer_2_biases.npy", bbml.Activations.linear))

pixelmatrix = np.zeros(784)
img_scale = 15
results = {}
empty = True
amount_predictions_shown = 5

running = True

while running:
    # event loop
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if clear_button_rect.collidepoint(pygame.mouse.get_pos()):
                    pixelmatrix = np.zeros(784)
                    empty = True
                elif center_button_rect.collidepoint(pygame.mouse.get_pos()):
                    if not empty:
                        pixelmatrix = center_drawing(pixelmatrix)

    # drawing logic
    if pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0] < img_scale * 28 - 1 and pygame.mouse.get_pos()[1] < img_scale * 28 - 1 and pygame.mouse.get_pos()[0] > 1 and pygame.mouse.get_pos()[1] > 1:
        empty = False
        matrix_location = (round(pygame.mouse.get_pos()[0] / img_scale) - 1, round(pygame.mouse.get_pos()[1] / img_scale) - 1)
        center_pix = cords_to_array(matrix_location)
        top_pix = cords_to_array((matrix_location[0], matrix_location[1] - 1))
        right_pix = cords_to_array((matrix_location[0] + 1, matrix_location[1]))
        bottom_pix = cords_to_array((matrix_location[0], matrix_location[1] + 1))
        left_pix = cords_to_array((matrix_location[0] - 1, matrix_location[1]))
        if center_pix >= 0 and center_pix < 784:
            pixelmatrix[center_pix] = 1.0
        if top_pix >= 0 and top_pix < 784:
            if pixelmatrix[top_pix] == 0.0:
                pixelmatrix[top_pix] = random.uniform(0.31, 0.78)
        if right_pix >= 0 and right_pix < 784:
            if pixelmatrix[right_pix] == 0.0:
                pixelmatrix[right_pix] = random.uniform(0.31, 0.78)
        if bottom_pix >= 0 and bottom_pix < 784:
            if pixelmatrix[bottom_pix] == 0.0:
                pixelmatrix[bottom_pix] = random.uniform(0.31, 0.78)
        if left_pix >= 0 and left_pix < 784:
            if pixelmatrix[left_pix] == 0.0:
                pixelmatrix[left_pix] = random.uniform(0.31, 0.78)

    # ai logic
    if not empty:
        prediction_array = network.forward(pixelmatrix)[0]
        softmax_prediction_array = softmax(prediction_array)
        for i in range(10):
            results[i] = softmaxed_prediction_array[i]
        sorted_results = [
            {"number": key, "confidence": value}
            for key, value in sorted(results.items(), key=lambda item: item[1], reverse=True)
        ]

    # displaying
    screen.fill((51, 51, 51))