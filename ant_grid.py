import pygame
import numpy as np
import matplotlib.pyplot as plt
from pygame.constants import MOUSEBUTTONDOWN

K1, K2 = 50, 50
N = 250

print('SHD max:', N*(K1*K2-N)/(K1*K2)**2)

config = 'RID' # RM, RID, AID

time_steps = 500
dt = 1
scale = 10

# TODO Functions
# Find the center of SFZ
# Find the area of a union of rectangles

class Ant:
    def __init__(self, location, task, walking_style, information):
        self.l = location
        self.p = task
        self.w = walking_style
        self.f = information
        self.beta = np.random.uniform(0,1)

        self.network = []

    def draw(self):
        color = (255*self.f, 0, 255*(1-self.f))
        i, j = self.l
        pygame.draw.rect(win, color, (i*scale, j*scale, scale, scale))

class Wall:
    def __init__(self, a, b):
        self.l = (min(a[0],b[0]), min(a[1],b[1]))
        self.size = (abs(b[0]-a[0]), abs(b[1]-a[1]))
    
    def draw(self):
        i, j = self.l
        w, h = self.size
        pygame.draw.rect(win, (0, 0, 0), (i*scale, j*scale, w*scale, h*scale))

class SFZ:
    '''
    Spatial Fidelity Zone (SFZ) on a grid of size K1 x K2.
    points: a set of all lattice points which are in the SFZ.
    '''
    def __init__(self, points, color = (100, 0, 100)):
        self.points = points
        self.color = color
        self.center = self.get_center()
    
    def dist(self, i, j):
        '''Returns the minimum L1 distance from (i,j) to the SFZ'''
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])
        return np.amin(np.absolute(x-i) + np.absolute(y-j))

    def get_center(self):
        return

    def draw(self):
        for p in self.points:
            pygame.draw.rect(win, self.color, (p[0]*scale, p[1]*scale, scale + 1, scale + 1), 1)

class Colony:
    '''
    K1, K2 : dimensions of the grid
    N : number of ants within the colony
    P: number of distinct tasks within in the colony
    '''
    def __init__(self, K1, K2, N, f, walls = [], sfzs = [], config = 'RM'):
        self.grid = np.zeros((K1, K2), dtype = int)
        self.ants = []
        self.sfzs = sfzs
        self.walls = []

        self.N = N
        self.f = f
        self.P = len(sfzs)
        self.Pl = np.zeros((K1, K2), dtype = int)
        
        self.contacts = np.array([])
        self.shd = np.array([])

    def get_sf(self, p):
        count = 0
        total = 0
        for ant in self.ants:
            if ant.p == p:
                total += 1
                if ant.w == 'D':
                    count += 1
        try:
            return count / total
        except:
            return None

    def get_shd(self):
        # TODO len(self.walls)
        return np.sum(np.square((self.Pl / len(self.contacts)) - (len(self.ants) / (K1*K2)))) / (K1*K2)

    def set_wall(self, start, stop):
        wall = Wall(start, stop)
        i, j = wall.l
        w, h = wall.size
        self.grid[i:i+w, j:j+h] = np.ones((w,h), dtype = int) * -1
        self.walls.append(wall)

    def create_ants(self):
        N = self.N
        while N > 0:
            i = np.random.randint(K1)
            j = np.random.randint(K2)
            if self.grid[i,j] == 0:
                if config == 'RM':
                    p = 0
                    w = 'R'
                elif config == 'RID':
                    p = np.random.randint(self.P)
                    w = np.random.choice(['R','D'], p = [1-self.f, self.f])
                elif config == 'AID':
                    p = np.random.randint(self.P)
                    w = np.random.choice(['R','D'], p = [1-self.f, self.f])
                ant = Ant((i,j), p, w, (N == 1))
                self.ants.append(ant)
                self.grid[i,j] = len(self.ants)
                self.Pl[i,j] += 1
                N -= 1

    def update(self):
        for ant in self.ants:
            i, j = ant.l
            self.Pl[i,j] += 1

        contacts = 0
        remaining_ants = list(range(len(self.ants)))
        while remaining_ants != []:
            n = remaining_ants.pop(np.random.randint(len(remaining_ants)))
            A = self.ants[n]
            i, j = A.l
            N = [(i+a, j+b) for a, b in [(1, 0), (-1, 0), (0, 1), (0, -1)] if min(i+a,j+b) > -1 and i+a < K1 and j+b < K2]
            N = [self.ants[self.grid[a,b]-1] for a, b in N if self.grid[a,b] not in [0, -1]]
            u1 = np.random.uniform(0, 1)
            if u1 > len(N) / 4:
                open_moves = [(i+a,j+b) for a, b in [(1, 0), (-1, 0), (0, 1), (0, -1)] if min(i+a,j+b) > -1 and i+a < K1 and j+b < K2]
                open_moves = [(a,b) for a, b in open_moves if self.grid[a,b] == 0]
                if A.w == 'R' and len(open_moves) > 0:
                    new_i, new_j = open_moves[np.random.choice(len(open_moves))]
                    A.l = (new_i, new_j)
                    self.grid[i, j], self.grid[new_i, new_j] = 0, n+1
                elif A.w == 'D' and len(open_moves) > 0:
                    sfz = self.sfzs[A.p]
                    min_dist = min([sfz.dist(a, b) for a, b in open_moves])
                    open_moves = [(a,b) for a, b in open_moves if sfz.dist(a, b) == min_dist]
                    new_i, new_j = open_moves[np.random.choice(len(open_moves))]
                    A.l = (new_i, new_j)
                    self.grid[i, j], self.grid[new_i, new_j] = 0, n+1
            else:
                B = N[np.random.choice(len(N))]
                u2 = np.random.uniform(0, 1)
                if u2 < B.beta and A.f != B.f:
                    contacts += 1
                    A.f, B.f = 1, 1
                    A.l, B.l = B.l, A.l
                    self.grid[A.l[0],A.l[1]], self.grid[B.l[0],B.l[1]] = self.grid[B.l[0],B.l[1]], self.grid[A.l[0],A.l[1]]

        if self.contacts.size == 0:
            self.contacts = np.append(self.contacts, [0])
        self.contacts = np.append(self.contacts, [self.contacts[-1] + contacts])
        self.shd = np.append(self.shd, self.get_shd())

    def draw(self):
        win.fill((255,255,255))
        # Draw grid lines (runs slow with high grid size)
        for i in range(K1):
            for j in range(K2):
                pygame.draw.line(win, (230,230,230), (i*scale, j*scale), (K1*scale, j*scale))
                pygame.draw.line(win, (230,230,230), (i*scale, j*scale), (i*scale, K2*scale))
        for n in range(len(self.ants)):
            self.ants[n].draw()
        for n in range(len(self.sfzs)):
            self.sfzs[n].draw()
        for n in range(len(self.walls)):
            self.walls[n].draw()
        pygame.display.update()

if __name__ == '__main__':
    # Start a new pygame window
    run = True
    start = None
    update = False
    win = pygame.display.set_mode((K1 * scale, K2 * scale))
    pygame.display.set_caption('Ant Nest Simulator')

    # Define SFZs
    sfzs = []
    colors = [(200, 0, 0), (200, 200, 0), (0, 200, 0), (0, 0, 200)]
    for n in range(3):
        i, j = np.random.randint(K1), np.random.randint(K2)
        sfzs.append(SFZ([(x,y) for x in range(i, i+3) for y in range(j, j+3)], colors[n]))

    f = [0.98, 0.8, 0.6, 0.4, 0.2]
    colonies = [Colony(K1, K2, N, f[i], sfzs = sfzs, config = config) for i in range(5)]
    
    t = 0
    clock = pygame.time.Clock()
    while run:
        clock.tick(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not update:
                cursor = list(pygame.mouse.get_pos())
                start = (cursor[0] // scale, cursor[1] // scale)
            elif event.type == pygame.MOUSEBUTTONUP and not update:
                cursor = list(pygame.mouse.get_pos())
                stop = (cursor[0] // scale + (cursor[0] // scale >= start[0]), cursor[1] // scale + (cursor[1] // scale >= start[1]))
                for colony in colonies:
                    colony.set_wall(start, stop)
                start = None
            elif event.type == pygame.KEYDOWN:
                update = True
                for colony in colonies:
                    if colony.ants == []:
                        colony.create_ants()

        colonies[0].draw()
        if start != None:
            cursor = list(pygame.mouse.get_pos())
            i, j = min(start[0], cursor[0] // scale), min(start[1], cursor[1] // scale)
            w, h = abs(cursor[0] // scale - start[0]) + (cursor[0] // scale >= start[0]), abs(cursor[1] // scale - start[1]) + (cursor[1] // scale >= start[1])
            pygame.draw.rect(win, (100, 100, 100), (i*scale, j*scale, w*scale, h*scale), 2)
            pygame.display.update()
            
        if update:
            for colony in colonies:
                colony.update()
            t += 1
            if t == time_steps:
                run = False

    SHDs = [colony.shd for colony in colonies]
    C = [colony.contacts for colony in colonies]
    R = np.array([(C[0][i+1]-C[0][i])/dt for i in range(len(C[0])-1)])
    #I = C / N

    # Spatial Fidelity of each task group
    '''for p in range(colony.P):
        print('SF('+str(p)+'):', colony.get_sf(p))'''

    plot = plt.figure(1)
    legend = []
    for i in range(len(SHDs)):
        plt.plot(SHDs[i])
        legend.append('SF:' + str(colonies[i].f))
    plt.title(config)
    plt.legend(legend)
    plt.xlabel('t')
    plt.ylabel('SHD')

    '''plot2 = plt.figure(2)
    plt.plot(I)
    plt.title('Proportion of Informed Ants')
    plt.xlabel('t')
    plt.ylabel('I(t)')

    plot3 = plt.figure(3)
    plt.scatter(np.arange(len(R)), R, s = 3)
    plt.title('Contact Rate')
    plt.xlabel('t')
    plt.ylabel('C(t) - C(t-dt)')'''

    plt.show()