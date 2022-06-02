import os

import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
from PIL import Image
from pygame import Surface

from neural_video.config import IMAGE_HEIGHT, IMAGE_WIDTH
from neural_video.model.basic import BasicVideoRenderer
from neural_video.model.renderer_base import VideoRenderer

pygame.init()
screen_main = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT))
clock = pygame.time.Clock()


def render_model(screen: Surface, model: VideoRenderer, angle: int) -> None:
    frame = model.generate_frame([angle])
    frame = frame.clip(0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.uint8)
    frame = Image.fromarray(frame)
    image = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode).convert()
    screen.blit(image, (0, 0))


def show_fps(screen: Surface, fps: str) -> None:
    font = pygame.font.SysFont("Calibri", 18)
    text = font.render(str(fps), True, pygame.Color((0, 255, 0)))
    screen.blit(text, (0, 0))


def run_rotating() -> None:
    model = BasicVideoRenderer()
    model.load_model(
        path_linear=os.path.join("results", "basic_linear.pt"),
        path_convolutional=os.path.join("results", "basic_convolutional.pt"),
    )
    frame_index_current = 0
    running = True
    while running:
        screen_main.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame_index_current += 1
        render_model(screen_main, model, frame_index_current)
        show_fps(screen_main, clock.get_fps())
        pygame.display.update()
        clock.tick(30)

        if frame_index_current > 144:
            running = False
