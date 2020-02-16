import googlemaps as gm
from googlemaps import distance_matrix
from operator import itemgetter

maps = gm.Client('(INSERT GOOGLE MAPS API KEY HERE)')

import matplotlib.pyplot as plt
import matplotlib.axes as axes

import numpy as np
from itertools import combinations
from docplex.mp.model import Model 
import math

import qiskit
from qiskit import Aer
from qiskit import BasicAer
from qiskit.tools.visualization import plot_histogram
from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.optimization.ising import docplex, max_cut, tsp
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.optimization.ising.common import sample_most_likely
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

import webbrowser

import logging
from qiskit.aqua import set_qiskit_aqua_logging

from qiskit import IBMQ
IBMQ.save_account('(INSERT IBMQ KEY HERE)', overwrite=True)
IBMQ.load_account()

import networkx as nx
# Ding the distance matrix
locations = ["135 Overlea Blvd, North York, ON M3C 1B3", 
"1040 Broadview Ave, Toronto, ON M4K 2S2",
"4 Astor Ave, East York, ON M4G 3M2",
"455 Cosburn Ave, East York, ON M4J 2N2",
"1500 Woodbine Ave, East York, ON M4C 4G9"]

coordinates = []
for i in locations:
    coordinates.append(list(maps.geocode(i)[0]['geometry']['location'].values()))

print(coordinates)
raw_matrix = maps.distance_matrix(coordinates, coordinates, mode='driving')
rows_dist = raw_matrix['rows']
print(rows_dist)

distance_matrix = []
for i in rows_dist:
    h = list(i.values())[0]
    list_thing = []
    for j in range(len(h)):
#         print(j, h[j]['distance'])
        if h[j]['distance']['text'].replace(' km', '') == '1 m':
            list_thing.append(0)
        else:
            list_thing.append(float(h[j]['distance']['text'].replace(' km', '')))
#         print(list_thing)
        if(j == 4):
            distance_matrix.append(list_thing)
    
# print(distance_matrix)
weight_matrix = np.array(distance_matrix)
print(weight_matrix)

print(coordinates)

# Create graph with random nodes
graph = nx.Graph(node_color = "red", alpha = 0.3)
nodes = len(coordinates) # Number of nodes
for node in range(len(coordinates)): # Add nodes onto graph
    graph.add_node(node)
pos = nx.spring_layout(graph)

# ins = tsp.random_tsp(len(coordinates))
# G = nx.Graph(coordinates)
# G.add_node(len(coordinates))

# print(ins)

nx.draw(graph)
print(pos)

# Create a graph
G = nx.Graph()

# distances
D = weight_matrix

labels = {}
for n in range(len(D)):
    for m in range(len(D)-(n+1)):
        G.add_edge(n,n+m+1)
        labels[ (n,n+m+1) ] = str(D[n][n+m+1])

pos=nx.spring_layout(G)

nx.draw(G, pos)
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size=10)
nx.draw_networkx_labels(G,pos=pos)

import pylab as plt
plt.show()

ins = tsp.random_tsp(len(coordinates))
qubitOp, offset = tsp.get_tsp_qubitops(ins)
algo_input = EnergyInput(qubitOp)

print(ins)

# Create an instance of a model and variables
mdl = Model(name='tsp')
x = {(i,p): mdl.binary_var(name='x_{0}_{1}'.format(i,p)) for i in range(n) for p in range(n)}

# Object function
tsp_func = mdl.sum(ins.w[i,j] * x[(i,p)] * x[(j,(p+1)%n)] for i in range(n) for j in range(n) for p in range(n))
mdl.minimize(tsp_func)

# Constrains
for i in range(n):
    mdl.add_constraint(mdl.sum(x[(i,p)] for p in range(n)) == 1)
for p in range(n):
    mdl.add_constraint(mdl.sum(x[(i,p)] for i in range(n)) == 1)

qubitOp_docplex, offset_docplex = docplex.get_qubitops(mdl)

#Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector
from qiskit.aqua.algorithms import VQE, ExactEigensolver
ee = ExactEigensolver(qubitOp, k=1)
result = ee.run()

"""
algorithm_cfg = {
    'name': 'ExactEigensolver',
}

params = {
    'problem': {'name': 'ising'},
    'algorithm': algorithm_cfg
}
result = run_algorithm(params,algo_input)
"""

print('energy:', result['energy'])
print('tsp objective:', result['energy'] + offset)
x = tsp.sample_most_likely(result['eigvecs'][0])
print('feasible:', tsp.tsp_feasible(x))
z = tsp.get_tsp_solution(x)
print('solution:', z)
print('solution objective:', tsp.tsp_value(z, ins.w))

# # Draw sol func
# def draw_tsp_solution(graph, order, colors, pos):
#     G2 = graph.copy()
#     nodes = len(order)
#     for i in range(nodes):
#         j = (i + 1) % nodes
#         G2.add_edge(order[i], order[j])
#     default_axes = plt.axes(frameon=True)
#     nx.draw_networkx(G2, node_color=red, node_size=600, alpha=.8, ax=default_axes, pos=pos)
# draw_tsp_solution(G, z, color=node_color, pos=pos)

print(z)
print(pos)

order = pos
latlong = coordinates

# Step 1 - sort lat values
latlong.sort(key=lambda tup: tup[0])

# Step 2 - sort order lat diff values
order = sorted(order.items(), key=lambda x: x[1][0])

# Step 3 - create dict linking sorted order with sorted latlong
linked = dict()
for i in range(len(latlong)):
  linked[order[i][0]] = latlong[i]

# Step 4 - sort and output results in visited order
ordered_coords = list()
for node_num in z:
  ordered_coords.append(latlong[node_num])
print("OUTPUT:", ordered_coords)


website = "https://www.google.com/maps/dir/"

for i in range(len(ordered_coords)):
    website += ordered_coords[i][0]
    website += ","
    website += ordered_coords[i][1]
    website += "/"

webbrowser.open_new(website)

# https://www.google.com/maps/dir/43.6978672,+-79.3671851/43.6842021,+-79.35706859999999/43.6919309,+-79.33564199999999/