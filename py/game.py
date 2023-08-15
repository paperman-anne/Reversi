import sys
import pygame

#pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("reversi")

running = True

while running:
    #poll for events
    #pygame.QUIT event means the user clicked x to close window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
           running = False

    #fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    #RENDER GAME HERE
    

    #flip the display to put work on screen
    pygame.display.flip()

pygame.quit()
sys.exit()