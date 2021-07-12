import pygame
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Ant:
    def __init__(self, name, location, task, walking_style, information):
        self.name = name
        self.l = location
        self.p = task
        self.w = walking_style
        self.f = information
        self.beta = np.random.uniform(0.01,0.1)

        self.network = []

    def draw(self, win, scale):
        color = (255*self.f, 0, 255*(1-self.f))
        i, j = self.l
        pygame.draw.rect(win, color, (i*scale, j*scale, scale, scale))

class Wall:
    def __init__(self, a, b):
        self.l = (min(a[0],b[0]), min(a[1],b[1]))
        self.size = (abs(b[0]-a[0]), abs(b[1]-a[1]))
    
    def draw(self, win, scale):
        i, j = self.l
        w, h = self.size
        pygame.draw.rect(win, (0, 0, 0), (i*scale, j*scale, w*scale, h*scale))

class Tunnel:
    def __init__(self, a, b):
        self.l = (min(a[0],b[0]), min(a[1],b[1]))
        self.size = (abs(b[0]-a[0]), abs(b[1]-a[1]))
    
    def draw(self, win, scale):
        i, j = self.l
        w, h = self.size
        pygame.draw.rect(win, (255, 255, 255), (i*scale, j*scale, w*scale, h*scale))
        for n in range(w+1):
            pygame.draw.line(win, (200, 200, 200), ((i+n)*scale, j*scale), ((i+n)*scale, (j+h)*scale))
        for n in range(h+1):
            pygame.draw.line(win, (200, 200, 200), (i*scale, (j+n)*scale), ((i+w)*scale, (j+n)*scale))

class SFZ:
    '''
    Spatial Fidelity Zone (SFZ) on a grid of size K1 x K2
    points: a set of all lattice points which are in the SFZ
    color: an RGB 255 color associated with the SFZ
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

    def intersect(self, sfz):
        return

    def draw(self, win, scale):
        for p in self.points:
            pygame.draw.rect(win, self.color, (p[0]*scale, p[1]*scale, scale + 1, scale + 1), 2)

class Colony:
    '''
    K1, K2 : dimensions of the grid
    N : number of ants within the colony
    f: spatial fidelity to assign each task
    P: number of SFZs within the colony
    config: initial distribution to setup colony with
    mode: determines whether additions to grid are additive or subtractive
    '''
    def __init__(self, K1, K2, N, f, P, config = 'RM', mode = 'wall'):
        self.grid = np.ones((K1, K2), dtype = int) * -1 * (mode == 'tunnel')
        self.ants = []
        self.sfzs = []
        self.walls = []
        self.tunnels = []

        self.N = N
        self.f = f
        self.P = P
        self.Pl = np.zeros((K1, K2), dtype = int)
        self.config = config
        self.mode = mode

        self.network = np.zeros((N, N), dtype = int)
        self.contacts = np.array([])
        self.shd = np.array([])

    def get_sf(self, p):
        '''Return the Spatial Fidelity of each task'''
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
        '''Return the Spatial Heterogeneity Degree of the colony'''
        K1, K2 = self.grid.shape
        return np.sum(np.square((self.Pl / len(self.contacts)) - (len(self.ants) / (K1*K2)))) / (K1*K2)

    def set_wall(self, start, stop):
        '''Create a Wall object between two points and append to self.walls'''
        wall = Wall(start, stop)
        i, j = wall.l
        w, h = wall.size
        self.grid[i:i+w, j:j+h] = np.ones((w,h), dtype = int) * -1
        self.walls.append(wall)

    def set_tunnel(self, start, stop):
        '''Create a Tunnel object between two points and append to self.tunnels'''
        tunnel = Tunnel(start, stop)
        i, j = tunnel.l
        w, h = tunnel.size
        self.grid[i:i+w, j:j+h] = np.zeros((w,h), dtype = int)
        self.tunnels.append(tunnel)

    def create_sfzs(self):
        '''Create P-many 3x3 sfzs which fit within the open spaces of the colony'''
        P = self.P
        K1, K2 = self.grid.shape
        colors = [(200, 0, 0), (200, 200, 0), (0, 200, 0)]
        while P > 0:
            i = np.random.randint(K1)
            j = np.random.randint(K2)
            if np.array_equal(self.grid[i-1:i+2,j-1:j+2], np.zeros((3,3))):
                print('done')
                sfz = SFZ([(x,y) for x in range(i-1,i+2) for y in range(j-1,j+2)], colors[len(colors) - P])
                #self.grid[i-1:i+2,j-1:j+2] = np.ones((3,3)) # Do this temporarily to avoid overlap
                self.sfzs.append(sfz)
                P -= 1
        # Reset the grid where we temporarily changed it
        for sfz in self.sfzs:
            for i, j in sfz.points:
                self.grid[i,j] = 0
        print(self.grid)

    def create_ants(self):
        '''Create ants based on rules associated with self.config'''
        N = self.N
        K1, K2 = self.grid.shape
        while N > 0:
            i = np.random.randint(K1)
            j = np.random.randint(K2)
            if self.grid[i,j] == 0:
                if self.config == 'RM':
                    p = 0
                    w = 'R'
                elif self.config == 'RID':
                    p = np.random.randint(self.P)
                    w = np.random.choice(['R','D'], p = [1-self.f, self.f])
                elif self.config == 'AID':
                    p = np.random.randint(self.P)
                    w = np.random.choice(['R','D'], p = [1-self.f, self.f])
                ant = Ant(len(self.ants)+1, (i,j), p, w, (N == 1))
                self.ants.append(ant)
                self.grid[i,j] = len(self.ants)
                self.Pl[i,j] += 1
                N -= 1

    def update(self):
        '''Make changes to colony for a single time step (described by flow chart in reference paper)'''
        K1, K2 = self.grid.shape
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
                if u2 < B.beta:
                    if A.f != B.f:
                        contacts += 1
                        A.f, B.f = 1, 1
                    A.l, B.l = B.l, A.l
                    self.network[A.name-1,B.name-1] += 1
                    self.grid[A.l[0],A.l[1]], self.grid[B.l[0],B.l[1]] = self.grid[B.l[0],B.l[1]], self.grid[A.l[0],A.l[1]]

        if self.contacts.size == 0:
            self.contacts = np.append(self.contacts, [0])
        self.contacts = np.append(self.contacts, [self.contacts[-1] + contacts])
        self.shd = np.append(self.shd, self.get_shd())

    def draw_grid(self, win, scale):
        '''Draw grid, walls or tunnels based on self.mode, ants, and sfzs'''
        K1, K2 = self.grid.shape
        if self.mode == 'wall':
            # Draw grid lines (runs slow with high grid size)
            win.fill((255, 255, 255))
            for i in range(K1):
                for j in range(K2):
                    pygame.draw.line(win, (230,230,230), (i*scale, j*scale), (K1*scale, j*scale))
                    pygame.draw.line(win, (230,230,230), (i*scale, j*scale), (i*scale, K2*scale))
            for n in range(len(self.walls)):
                self.walls[n].draw(win, scale)
        elif self.mode == 'tunnel':
            win.fill((0, 0, 0))
            for n in range(len(self.tunnels)):
                self.tunnels[n].draw(win, scale)
        
        for n in range(len(self.ants)):
            self.ants[n].draw(win, scale)
        for n in range(len(self.sfzs)):
            self.sfzs[n].draw(win, scale)
        pygame.display.update()

    def draw_network(self):
        '''Display directed connections between ants with color representing walking style / task zone'''
        #print(self.network)
        color_map = []
        for ant in self.ants:
            if ant.w == 'R':
                color = (0,0,0)
            else:
                color = self.sfzs[ant.p].color
                color = (color[0]/255, color[1]/255, color[2]/255)
            color_map.append(color)
        G = nx.from_numpy_matrix(self.network, parallel_edges = True, create_using = nx.DiGraph())
        nx.draw(G, with_labels = True, node_size = 1500, node_color = color_map, alpha = 0.5, arrows = True)
        plt.show()

    def get_motifs(self):
        return