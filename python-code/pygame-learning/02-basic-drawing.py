#coding:utf-8

import pygame
import sys
from pygame.locals import *

pygame.init()
surface = pygame.display.set_mode((600,480))
pygame.display.set_caption('Hello Wrold')

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit(0)

    pygame.draw.line(surface,(0,0,255),(60,60),(200,60),4)
    pygame.draw.circle(surface,(255,0,0),(60,60),50)
    pygame.draw.circle(surface,(0,255,0),(110,110),50,0)
    
    pygame.display.update()