import pygame
import numpy as np
import sys

pygame.init()

size = width, height = 640, 480
speed = [8, 4]
black = (0, 0, 0)
white = (255, 255, 255)

fps_time = 30

circle_position = [200, 200]
circle_radius = 40

screen = pygame.display.set_mode(size)

pygame.display.set_caption("moving circle")

clock = pygame.time.Clock()

is_running = True

while is_running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(black)

    circle_position[0] = circle_position[0] + speed[0]
    circle_position[1] = circle_position[1] + speed[1]

    pygame.draw.circle(screen, white, circle_position, circle_radius)

    if (circle_position[0]+speed[0]) > (width - circle_radius):
        speed[0] = -speed[0]

    if (circle_position[1]+speed[1]) > (height - circle_radius):
        speed[1] = -speed[1]

    if (circle_position[0]+speed[0]) < (0 + circle_radius):
        speed[0] = -speed[0]

    if (circle_position[1]+speed[1]) < (0 + circle_radius):
        speed[1] = -speed[1]

    clock.tick(fps_time)
    pygame.display.flip()

