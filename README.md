# Social Insects
This is the code repository for the social insect group at the Arizona State University QRLSSP Program. Included are several different simulations of ant colonies and their interaction networks. Some tools for analysis are built into each of the programs with detailed information about each below.

## ant_grid
This program is taken from the paper 'Dynamics of social interactions, in the flow of information and disease spreading in social insects colonies: Effects of environmental events and spatial heterogeneity.' In this paper, ants are modeled on a grid following different movement patterns and having assigned tasks. Some ants will tend to move toward their assigned task zone while others move randomly. The simulation is meant to study the presence and specific shapes and locations of these task zones on the spread of an agent (either information or a pathogen) through the colony over time.

**How to use:**


## ant_nest
Many studies have been done on interaction networks above ground, but there is a distinct lack of simulations for ants within their nest. This program models the nest as a random undirected graph with a given number of nodes and edges and places ants to move along these edges. Each ant has a random unlarmed speed which increases by a random amount when it becomes alarmed by another ant. Interactions between ants are recorded and displayed both on top of the nest and to the side in a static fashion.
