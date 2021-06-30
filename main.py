import pygame
import numpy as np

window_size = 500

class Ant:
    '''Ants have a label, position, speed, excitation state, and interaction radius
        label : a unique integer
        position: an (x,y) coordinate which correspond to a location in the nest
        speed: a value between 0 and 1 representing what percentage of an edge the ant will travel along in an amount of time
        excited: a boolean to mark alarmed state
        radius: an integer to determine if ants transfer information'''
    def __init__(self, label, position, speed, excited = False, radius = 5):
        self.label = label
        self.position = position
        self.speed = speed
        self.excited = excited
        self.radius = radius

    def draw(self):
        if self.excited:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        pygame.draw.circle(win, color, self.position, 5)

class Nest:
    '''Nests have a graph structure which can be either directed or undirected'''
    def __init__(self, nodes = [], edges = [], directed = True):
        self.graph = {}
        self.directed = directed
        for node in nodes:
            self.graph[node] = set([])
        for edge in edges:
            self.graph[edge[0]].add(edge[1])
            if not directed:
                self.graph[edge[1]].add(edge[0])

        self.ants = []
    
    def randomize_graph(self, nodes, edges):
        '''Create a random graph with a given number of nodes and edges'''

        # Set upper bound on edges so we don't run forever
        if self.directed:
            edges = min(edges, edges * (edges - 1))
        else:
            edges = min(edges, edges * (edges - 1) / 2)

        # Clear dictionary
        self.graph = {}

        # Create nodes
        for n in range(nodes):
            self.graph[n] = set([])

        # Create random edges
        edge_buffer = []
        while edges > 0:
            a = np.random.randint(nodes)
            b = np.random.randint(nodes)
            edge = (a, b)
            if edge not in edge_buffer:
                edge_buffer.append(edge)
                edges -= 1
        
        # Add edges to dictionary
        for edge in edge_buffer:
            self.graph[edge[0]].add(edge[1])
            if not self.directed:
                self.graph[edge[1]].add(edge[0])

    def randomize_ants(self, num):
        '''Chooses a randomly location along an edges and places an ant there'''
        
        # Clear ant list
        self.ants = []

        # Create random ants
        for n in range(num):
            # Choose which nodes ant will lie between
            node1 = np.random.randint(len(self.graph))
            print(self.graph[node1])
            node2 = np.random.choice(list(self.graph[node1]))

            # Choose how far along the edge the ant is
            distance = np.random.uniform()

            # Calculate the position of the ant along the edge
            theta = 2 * np.pi / len(self.graph)
            x1 = int(window_size / 2 + 0.3 * window_size * np.cos(node1 * theta))
            y1 = int(window_size / 2 + 0.3 * window_size * np.sin(node1 * theta))
            x2 = int(window_size / 2 + 0.3 * window_size * np.cos(node2 * theta))
            y2 = int(window_size / 2 + 0.3 * window_size * np.sin(node2 * theta))
            x_dist = x2 - x1
            y_dist = y2 - y1

            # Create and append ant to self.ants
            self.ants.append(Ant(n, (x1 + distance * x_dist, y1 + distance * y_dist), 0.05))

    def draw(self):
        '''Draw evenly spaced nodes on a circle with their connections to show the structure of the nest'''
        # Draw nodes
        theta = 2 * np.pi / len(self.graph)
        for n in range(len(self.graph)):
            x = window_size / 2 + 0.3 * window_size * np.cos(n * theta)
            y = window_size / 2 + 0.3 * window_size * np.sin(n * theta)
            pygame.draw.circle(win, (0,0,0), (x, y), 10)

        # Draw edges
        for node1 in range(len(self.graph)):
            for node2 in self.graph[node1]:
                x1 = int(window_size / 2 + 0.3 * window_size * np.cos(node1 * theta))
                y1 = int(window_size / 2 + 0.3 * window_size * np.sin(node1 * theta))
                x2 = int(window_size / 2 + 0.3 * window_size * np.cos(node2 * theta))
                y2 = int(window_size / 2 + 0.3 * window_size * np.sin(node2 * theta))
                pygame.draw.line(win, (0,0,0), (x1, y1), (x2, y2), 2)
                if node1 == node2:
                    pygame.draw.circle(win, (0,0,0), (x1 + 20 * np.cos(node1 * theta), y1 + 20 * np.sin(node1 * theta)), 20, 2)
        
        for ant in self.ants:
            ant.draw()

if __name__ == '__main__':
    win = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption('Ant Nest Simulator')
    win.fill((255,255,255))

    n = Nest(directed = False)
    n.randomize_graph(6, 8)
    n.randomize_ants(10)

    print(n.graph)
    n.draw()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        pygame.display.update()