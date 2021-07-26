import glob
import pygame
import numpy as np

current_folder = 'git/social-insects/results/' # Change depending on your local setup

view = 0 # which control we are currently viewing
plane = 0
fig = 0

scale = 300 # height of a single image

shapes = ['s1', 's2', 's3']
densities = ['d0.1', 'd0.2', 'd0.3']
sfs = ['sf0.2', 'sf0.5', 'sf0.8']
figures = ['avgi', 'i', 'motifs', 'rb', 'rw', 'shd']

# Defaults
structure = 'Chains'
figure = 'avgi'
control = shapes
rows = densities
cols = sfs

debug = False

if __name__ == '__main__':
    win = pygame.display.set_mode((scale*3 + 60, int(scale*2.25) + 60))
    pygame.display.set_caption('Ant Nest Simulator')
    pygame.font.init()
    font = pygame.font.SysFont('Calibri', 25)
    font2 = pygame.font.SysFont('Calibri', 20)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            
            elif event.type == pygame.KEYDOWN:
                # print(event.key)
                if event.key == pygame.K_UP:
                    view = min(2, view + 1)
                elif event.key == pygame.K_DOWN:
                    view = max(0, view - 1)
                elif event.key == pygame.K_LEFT:
                    plane = (plane + 1) % 3
                elif event.key == pygame.K_RIGHT:
                    plane = (plane - 1) % 3
                # (s)quares, (d)onuts, (t)unnels, (c)hains
                elif event.key in [115, 100, 116, 99]:
                    structure = ['Squares', 'Donuts', 'Tunnels', 'Chains'][[115, 100, 116, 99].index(event.key)]
                elif event.key in [49, 50, 51, 52, 53, 54]:
                    fig = event.key - 49
                # s(h)apes, d(e)nsities, s(f)s
                '''elif event.key in [104, 102, 101]:
                    options = [shapes, sfs, densities]
                    control = options.pop([104, 102, 101].index(event.key))
                    rows, cols = options'''
        
        options = [shapes, sfs, densities]
        control = options[plane]
        rows, cols = options[(plane + 1) % 3], options[(plane + 2) % 3]

        # Get images
        files = [file for file in glob.glob(current_folder + structure + '/' + figures[fig] + '/' + '*.png') if control[view] in file]
        
        if debug:
            print()
            print('control', control[view])
            print('rows', rows)
            print('files', len(files))
            print(files)

        images = []
        for row in rows:
            temp_row = [file for file in files if row in file]
            images.append([pygame.transform.scale(pygame.image.load(file), (scale, int(scale*0.75))) for file in temp_row])
            
            if debug:
                print('temp rows', len(temp_row))

        # Draw images
        win.fill((255, 255, 255))
        for i in range(3):
            for j in range(3):
                win.blit(images[i][j], (scale*i + 30, int(scale*.75*j) + 30))

        # Shape with control
        text = font.render(structure + ' (' + control[view] + ')', False, (0, 0, 0))
        text_rect = text.get_rect(center = (scale*1.5 + 30, 25))
        win.blit(text, text_rect)

        # Row variable
        for i in range(3):
            text = font2.render(rows[i], False, (0, 0, 0))
            text_rect = text.get_rect(center = (20, 30 + scale*2.25*(2*i+1)/6))
            win.blit(text, text_rect)

        # Column variable
        for i in range(3):
            text = font2.render(cols[i], False, (0, 0, 0))
            text_rect = text.get_rect(center = (scale*(2*i+1)/2 + 30, scale*2.25 + 40))
            win.blit(text, text_rect)

        pygame.display.update()