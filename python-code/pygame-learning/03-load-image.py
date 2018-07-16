#coding:utf-8

import pygame
import sys
from pygame.locals import *

pygame.init()
surface = pygame.display.set_mode((640,480),0,32)
pygame.display.set_caption('Hello Wrold')
WHITE = (255,255,255)
image = pygame.image.load('datas/game-resources/smile_128x128.png')
cx,cy = 10,10
direction = 'right'
while True:
    surface.fill(WHITE)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit(0)
    if direction == 'right':
        cx += 5
        if cx >= 512:
            direction = 'down'
    elif direction == 'down':
        cy += 5
        if cy >= 352:
            direction = 'left'
    elif direction == 'left':
        cx -= 5
        if cx <= 0:
            direction = 'up'
    elif direction == 'up':
        cy -= 5
        if cy <= 0:
            direction = 'right'
    surface.blit(image,(cx,cy))
    pygame.display.update()