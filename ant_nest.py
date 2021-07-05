import time
import pygame
import numpy as np

window_size = 500

class Ant:
    '''Ants have a label, position, speed, excitation state, and interaction radius
        label : a unique integer
        node1, node2: integer pair of indecies of nodes the ant is traveling between (from first to second)
        distance: an decimal length along the edge the ant is currently at
        speed: a value between 0 and 1 representing what percentage of an edge the ant will travel along in an amount of time
        excited: a boolean to mark alarmed state
        radius: an integer to determine if ants transfer information'''
    def __init__(self, label, nest, node1, node2, distance, speed, alarmed = False, radius = 5):
        self.label = label

        self.nest = nest
        self.node1 = node1
        self.node2 = node2

        self.distance = distance
        self.speed = speed
        self.alarmed = alarmed
        self.radius = radius

    def move(self):
        self.distance += self.speed / self.get_length()
        if self.distance > 0.98:
            self.node1 = self.node2
            if list(self.nest.graph[self.node2]) == []:
                self.node1, self.node2 = self.node2, self.node1
            else:
                self.node1 = self.node2
                self.node2 = np.random.choice(list(self.nest.graph[self.node2]))
            self.distance = 0

    def get_position(self):
        theta = 2 * np.pi / len(self.nest.graph)
        x1 = int(window_size / 2 + 0.3 * window_size * np.cos(self.node1 * theta))
        y1 = int(window_size / 2 + 0.3 * window_size * np.sin(self.node1 * theta))
        x2 = int(window_size / 2 + 0.3 * window_size * np.cos(self.node2 * theta))
        y2 = int(window_size / 2 + 0.3 * window_size * np.sin(self.node2 * theta))
        x, y = x2 - x1, y2 - y1
        return x1 + x * self.distance, y1 + y * self.distance
    
    def get_length(self):
        theta = 2 * np.pi / len(self.nest.graph)
        x1 = int(window_size / 2 + 0.3 * window_size * np.cos(self.node1 * theta))
        y1 = int(window_size / 2 + 0.3 * window_size * np.sin(self.node1 * theta))
        x2 = int(window_size / 2 + 0.3 * window_size * np.cos(self.node2 * theta))
        y2 = int(window_size / 2 + 0.3 * window_size * np.sin(self.node2 * theta))
        return np.sqrt((x2 - x1)**2 + (y2- y1)**2)

    def draw(self):
        if self.alarmed:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        pygame.draw.circle(win, color, (self.get_position()), 5)

class AntNetwork:
    def __init__(self, ants, directed = True):
        self.graph = {}
        self.directed = directed
        for ant in ants:
            self.graph[ant] = set([])

    def add_connection(self, ant1, ant2):
        self.graph[ant1].add(ant2)
        if not self.directed:
            self.graph[ant2].add(ant1)
    
    def get_alarmed(self):
        '''Returns the number of alarmed ants'''
        total = 0
        for ant in list(self.graph.keys()):
            total += ant.alarmed
        return total

    def draw(self):
        # Draw ant network on top of nest
        for ant1 in self.graph:
            for ant2 in self.graph[ant1]:
                if ant1.alarmed or ant2.alarmed:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)
                pygame.draw.line(win, color, (ant1.get_position()), (ant2.get_position()), 1)

        # Draw static ant network to the side
        theta = 2*np.pi / len(self.graph)
        ants = list(self.graph.keys())
        for n in range(len(self.graph)):
            if ants[n].alarmed:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            x1 = window_size + int(window_size / 2 + 0.3 * window_size * np.cos(n * theta))
            y1 = int(window_size / 2 + 0.3 * window_size * np.sin(n * theta))
            pygame.draw.circle(win, color, (x1, y1), 5)
            connected_ants = list(self.graph[ants[n]])
            for k in range(len(connected_ants)):
                k = ants.index(connected_ants[k])
                x2 = window_size + int(window_size / 2 + 0.3 * window_size * np.cos(k * theta))
                y2 = int(window_size / 2 + 0.3 * window_size * np.sin(k * theta))
                pygame.draw.line(win, color, (x1, y1), (x2, y2), 1)
                
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
        self.ant_network = AntNetwork(self.ants)
    
    def interact(self):
        checks = 0
        # Check every combination of ants
        '''for ant1 in self.ants:
            for ant2 in self.ants:
                if ant1 != ant2:
                    ant1x, ant1y = ant1.get_position()
                    ant2x, ant2y = ant2.get_position()
                    if np.sqrt((ant2x - ant1x)**2 + (ant2y - ant1y)**2) < ant1.radius:
                        self.ant_network.add_connection(ant1, ant2)
                        if ant1.alarmed and not ant2.alarmed:
                            ant2.alarmed = True
                            ant2.speed += np.random.uniform(0.03, 0.07)
                    checks += 1'''
        # Check every combination of ants
        '''for i in range(len(self.ants)):
            for j in range(i+1, len(self.ants)):
                ant1, ant2 = self.ants[i], self.ants[j]
                ant1x, ant1y = ant1.get_position()
                ant2x, ant2y = ant2.get_position()
                if np.sqrt((ant2x - ant1x)**2 + (ant2y - ant1y)**2) < ant1.radius:
                    self.ant_network.add_connection(ant1, ant2)
                    if ant1.alarmed and not ant2.alarmed:
                        ant2.alarmed = True
                        ant2.speed += np.random.uniform(0.03, 0.07)
                    if ant2.alarmed and not ant1.alarmed:
                        ant1.alarmed = True
                        ant1.speed += np.random.uniform(0.03, 0.07)
                checks += 1'''
        # Check ants which are on the same edge
        for i in range(len(self.ants)):
            for j in range(i+1, len(self.ants)):
                ant1, ant2 = self.ants[i], self.ants[j]
                if set([ant1.node1, ant1.node2]) == set([ant2.node1, ant2.node2]):                    
                    ant1x, ant1y = ant1.get_position()
                    ant2x, ant2y = ant2.get_position()
                    if np.sqrt((ant2x - ant1x)**2 + (ant2y - ant1y)**2) < ant1.radius:
                        self.ant_network.add_connection(ant1, ant2)
                        if ant1.alarmed and not ant2.alarmed:
                            ant2.alarmed = True
                            ant2.speed += np.random.uniform(0.03, 0.07)
                        if ant2.alarmed and not ant1.alarmed:
                            ant1.alarmed = True
                            ant1.speed += np.random.uniform(0.03, 0.07)
                    checks += 1
        # print('checks', checks)

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
            node2 = np.random.choice(list(self.graph[node1]))

            # Choose where the ant will lie along the edge
            distance = np.random.uniform()
            
            # Create and append ant to self.ants
            alarmed = (n == num - 1) 
            #alarmed = False
            self.ants.append(Ant(n, self, node1, node2, distance, np.random.uniform(0.05 + 0.05*alarmed, 0.07 + 0.05*alarmed), alarmed))
        
        self.ant_network = AntNetwork(self.ants)

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
        
        self.ant_network.draw()
        for ant in self.ants:
            ant.draw()

if __name__ == '__main__':
    win = pygame.display.set_mode((window_size*2, int(window_size*1.5)))
    pygame.display.set_caption('Ant Nest Simulator')
    win.fill((255,255,255))

    nest = Nest(directed = False)

    nest.randomize_graph(5, 7)
    nest.randomize_ants(5)

    '''nest.randomize_graph(10, 30)
    nest.randomize_ants(30)'''

    '''nest.randomize_graph(8, 10)
    nest.randomize_ants(50)'''

    print(nest.graph)

    # Track alarmed ants over time
    alarmed_ant_total = nest.ant_network.get_alarmed() / len(nest.ants)
    data_points = [(0, alarmed_ant_total)]

    frame = 0
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        win.fill((255,255,255))

        # Timing Stuff for Efficiency Testing
        '''start_draw = time.time()
        nest.draw()
        end_draw = time.time()
        
        start_move = time.time()
        for ant in nest.ants:
            ant.move()
        end_move = time.time()

        start_interact = time.time()
        nest.interact()
        end_interact = time.time()

        if frame % 1000 == 0:
            print(frame)
            print('Draw Time:', end_draw - start_draw, 'Move Time:', end_move - start_move, 'Interact Time:', end_interact - start_interact)
            print('Frame Time:', end_draw - start_draw + end_move - start_move + end_interact - start_interact)
            print()'''
        
        nest.draw()
        for ant in nest.ants:
            ant.move()
        nest.interact()

        # Draw graph of alarmed ants over time
        if nest.ant_network.get_alarmed() > alarmed_ant_total:
            alarmed_ant_total = nest.ant_network.get_alarmed()
            data_points.append((frame, alarmed_ant_total / len(nest.ants)))

        # Axes
        pygame.draw.line(win, (0,0,0), (window_size * 0.3, window_size * 1.3), (min(int(window_size * 0.3 + frame / 35), window_size * 1.7), window_size * 1.3), 2)
        pygame.draw.line(win, (0,0,0), (window_size * 0.3, window_size * 0.9), (window_size * 0.3, window_size * 1.3), 2)
        pygame.draw.line(win, (255,0,0), (int(window_size * 0.3 + data_points[-1][0] / 35), window_size * 1.3 - data_points[-1][1] * window_size * 0.4), (min(int(window_size * 0.3 + frame / 35), window_size * 1.7), window_size * 1.3 - data_points[-1][1] * window_size * 0.4), 1)
        
        # Plot
        for n in range(len(data_points)):
            pygame.draw.circle(win, (255,0,0), (int(window_size * 0.3 + data_points[n][0] / 35), window_size * 1.3 - data_points[n][1] * window_size * 0.4), 2)
            if n != len(data_points) - 1:
                #pygame.draw.line(win, (255,0,0), (int(window_size * 0.3 + data_points[n][0] / 35), window_size * 1.3 - data_points[n][1] * window_size * 0.4), (int(window_size * 0.3 + data_points[n+1][0] / 35), window_size * 1.3 - data_points[n+1][1] * window_size * 0.4), 1)
                pygame.draw.line(win, (255,0,0), (int(window_size * 0.3 + data_points[n][0] / 35), window_size * 1.3 - data_points[n][1] * window_size * 0.4), (int(window_size * 0.3 + data_points[n+1][0] / 35), window_size * 1.3 - data_points[n][1] * window_size * 0.4), 1)
                pygame.draw.line(win, (255,0,0), (int(window_size * 0.3 + data_points[n+1][0] / 35), window_size * 1.3 - data_points[n][1] * window_size * 0.4), (int(window_size * 0.3 + data_points[n+1][0] / 35), window_size * 1.3 - data_points[n+1][1] * window_size * 0.4), 1)

        pygame.display.update()

        frame += 1