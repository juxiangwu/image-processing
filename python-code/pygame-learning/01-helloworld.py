#coding:utf-8

import pygame
import sys
from pygame.locals import *

pygame.init()
diplaysurf = pygame.display.set_mode((600,480))
pygame.display.set_caption('Hello Wrold')

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit(0)
    pygame.display.update()