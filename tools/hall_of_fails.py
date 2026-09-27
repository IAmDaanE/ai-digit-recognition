import json
from utils import load_mnist_csv, array_to_cords
import pygame

pygame.init()
screen = pygame.display.set_mode((840, 522))
pygame.display.set_caption("AI Model Diagnostics")
clock = pygame.time.Clock()

images, labels = load_mnist_csv("../data/mnist_test.csv")

model_name = ""
fails = []
with open("../outputs/fails.json", "r") as f:
    raw_unloaded_data = json.load(f)
    model_name = raw_unloaded_data["model"]
    fails = raw_unloaded_data["fails"]

cur_fail_id = 0
img_scale = 15
amount_fails = len(fails)

window_width = 840
window_height = 522
padding = 20
button_height = 42
button_width = (window_width - padding * 4) / 3
diag_width = 360

diag_rect = pygame.Rect(padding * 2 + 420, padding, 360, 420)

left_button_rect = pygame.Rect(padding, padding * 2 + 420, button_width, button_height)
right_button_rect = pygame.Rect(padding * 3 + button_width * 2, padding * 2 + 420, button_width, button_height)
page_text_rect = pygame.Rect(padding * 2 + button_width, padding * 2 + 420, button_width, button_height)

button_font = pygame.font.Font("../assets/Poppins-Regular.ttf", 30)
page_text_font = pygame.font.Font("../assets/Poppins-Bold.ttf", 24)
info_font = pygame.font.Font("../assets/Poppins-Regular.ttf", 20)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            if left_button_rect.collidepoint(pos):
                if cur_fail_id > 0:
                    cur_fail_id -= 1
            elif right_button_rect.collidepoint(pos):
                if cur_fail_id < amount_fails - 1:
                    cur_fail_id += 1
    
    screen.fill((51, 51, 51))
    for i, color in enumerate(images[fails[cur_fail_id]["image_id"]]):
        x, y = array_to_cords(i)
        rgb_color = (round(color * 255), round(color * 255), round(color * 255))
        pygame.draw.rect(screen, rgb_color, (padding + x * img_scale, padding + y * img_scale, img_scale, img_scale))
    pygame.draw.rect(screen, (0, 173, 181), (20, 20, 420, 420), width=1)

    pygame.draw.rect(screen, (0, 0, 0), diag_rect)
    pygame.draw.rect(screen, (0, 173, 181), diag_rect, width=1)
    model_guess_text = info_font.render(f"Model Guess: {fails[cur_fail_id]['guessed']}", True, (255, 255, 255))
    correct_label_text = info_font.render(f"Correct Label: {fails[cur_fail_id]['actual']}", True, (255, 255, 255))
    confidence_text = info_font.render(f"Confidence: {fails[cur_fail_id]['confidence']}%", True, (255, 255, 255))
    model_name_text = info_font.render(f"Model Name: {model_name}", True, (255, 255, 255))
    screen.blit(model_guess_text, (diag_rect.x + 12, padding + 4))
    screen.blit(correct_label_text, (diag_rect.x + 12, model_guess_text.height + padding + 10))
    screen.blit(confidence_text, (diag_rect.x + 12, correct_label_text.height + model_guess_text.height + 16 + padding))
    screen.blit(model_name_text, (diag_rect.x + 12, 402))

    pygame.draw.rect(screen, (0, 173, 181), left_button_rect)
    left_text = button_font.render("<<", True, (0, 0, 0))
    left_text_rect = left_text.get_rect(center=left_button_rect.center)
    screen.blit(left_text, left_text_rect)
    
    pygame.draw.rect(screen, (0, 173, 181), right_button_rect)
    right_text = button_font.render(">>", True, (0, 0, 0))
    right_text_rect = right_text.get_rect(center=right_button_rect.center)
    screen.blit(right_text, right_text_rect)

    pygame.draw.rect(screen, (0, 0, 0), page_text_rect)
    pygame.draw.rect(screen, (0, 173, 181), page_text_rect, width=1)
    page_text_text = page_text_font.render(f"{cur_fail_id + 1}/{amount_fails}", True, (255, 255, 255))
    page_text_text_rect = page_text_text.get_rect(center=page_text_rect.center)
    screen.blit(page_text_text, page_text_text_rect)

    pygame.display.update()