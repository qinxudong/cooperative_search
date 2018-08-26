#coding=utf-8
from agent import Agent
import numpy as np
import pandas as pd

#输入搜索区域矩阵数据
df = pd.read_csv('/home/qin/github/multiagent_targetsearch-qin/input1.csv', header=None)
matrix_sr = df.values

#参数赋值
num_row = matrix_sr.shape[0]
num_column = matrix_sr.shape[1]
map_init = np.zeros([num_row, num_column])
coords = [[2, 3], [6, 9], [12, 7], [18, 22], [16, 5], [13, 8]]
p = 0.9
q = 0.3
radius_search = 6
radius_communicate = 10
radius_consider = 8
bound_speed = 3
bound_Q = 10000000000000000

#agent初始化
agent0 = Agent(0, coords[0], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,  bound_Q)
agent1 = Agent(1, coords[1], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,  bound_Q)
agent2 = Agent(2, coords[2], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,  bound_Q)
agent3 = Agent(3, coords[3], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,  bound_Q)
agent4 = Agent(4, coords[4], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,  bound_Q)
agent5 = Agent(5, coords[5], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,  bound_Q)

#初始化一些后面用到的列表
list_agent = [agent0, agent1, agent2, agent3, agent4, agent5]#agent列表
N = len(list_agent)#agent数量
count = 0#循环次数

#主循环
while True:
    #单个agent概率图更新
    for agent in list_agent:
        cell_search = agent.cellsearch()
        agent.update(matrix_sr, cell_search)

    '''
    coord_all = []  # 所有agent坐标的列表
    list_map = []  # 所有agent概率图的列表
    for agent in list_agent:
        coord_all.append(agent.coord)
        list_map.append(agent.map)
    array_map = np.array(list_map)  # 转array
    '''
    #概率图融合
    for agent in list_agent:
        agent_neighbor = agent.neighbor(coord_all)
        agent.fuse(agent_neighbor, N, array_map)

    #agent移动
    for agent in list_agent:
        cell_consider = agent.cellconsider()
        agent.move(cell_consider)

    count += 1
    print(count)

    #循环退出条件
    unsure = 0#概率图的不确定度
    for agent in list_agent:
        for i in range(num_row):
            for j in range(num_column):
                unsure += np.exp(-2*np.linalg.norm(agent.map[i][j]))
    unsure_final = unsure / (N * num_column * num_row)
    if unsure_final <= 0.01:
        for agent in list_agent:
            maps = []
            maps.append(float(1)/(np.exp(agent.map) + 1))
            print(maps)
        #输出
        output = pd.DataFrame(maps[0])
        output.to_csv('output.csv', index=False)

        break
