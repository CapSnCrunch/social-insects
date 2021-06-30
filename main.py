import pygame
import numpy as np

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
    
    def randomize(self, nodes, edges):
        '''Create a random graph with a given number of nodes and edges'''

        # Set upper bound on edges so we don't run forever
        if self.directed:
            edges = min(edges, edges * (edges - 1))
        else:
            edges = min(edges, edges * (edges - 1) / 2)

        # Clear dictionary
        self.graph = {}

        # Create nodes
        for i in range(nodes):
            self.graph[i] = set([])

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

if __name__ == '__main__':

    nodes = [1, 2, 3]
    edges = [(1, 2), (1, 3)]

    n = Nest(nodes, edges, directed = True)
    n.randomize(5, 5)

    print(n.graph)