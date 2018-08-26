from agent import *
import pandas as pd
import numpy as np

df = pd.read_csv('/home/qin/github/multiagent_targetsearch-qin/test_input.csv', header=None)
matrix_sr = df.values
num_row = matrix_sr.shape[0]
num_column = matrix_sr.shape[1]
map_init = np.zeros([num_row, num_column])
agent = Agent(index=0, coord=(3,3), map=map_init, p=0.9, q=0.3, radius_search=8, radius_communicate=10, radius_consider=15, bound_speed=3,  bound_Q=1000000000000000)

while True:
    cell_search = agent.cellsearch()
    agent.update(matrix_sr, cell_search)
    agent.fuse(agent_neighbor=[1,], N=1, array_map)
    cell_consider = agent.cellsearch()
    agent.move(cell_consider)
    print(agent.map)