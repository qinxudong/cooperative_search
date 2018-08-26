#coding=utf-8
from agent import Agent
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#输入搜索区域矩阵数据
df = pd.read_csv('/home/qin/code/experiments/cooperative_search/input2.csv', header=None)
matrix_sr = df.values

#参数定义
num_row = matrix_sr.shape[0]
num_column = matrix_sr.shape[1]
map_init = np.zeros([num_row, num_column])
coords = [[2, 3], [6, 9], [12, 7], [18, 22], [16, 5], [13, 8]]
p = 0.9
q = 0.3
radius_search = 4
radius_communicate = 6
radius_consider = 6
bound_speed = 2
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

#热图初始化
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
map_im = float(1)/(np.exp(map_init) + 1)
im = ax.imshow(map_im, cmap='bone_r', vmin=0, vmax=1)
plt.colorbar(im, shrink=0.5)
coord_pl = [[], []]
coord_pl[0].append(coords[0][0])
coord_pl[1].append(coords[0][1])
pl = ax.plot(coord_pl[0], coord_pl[1], color='red', marker='>', linewidth=0.3)
plt.pause(20)

#主循环
while True:

    #单个agent概率图更新
    for agent in list_agent:
        cell_search = agent.cellsearch()
        agent.update(matrix_sr, cell_search)
    #概率图融合
    coord_all = []  # 所有agent坐标的列表
    list_map = []  # 所有agent概率图的列表
    for agent in list_agent:
        coord_all.append(agent.coord)
        list_map.append(agent.map)
    array_map = np.array(list_map)  # 转array
    for agent in list_agent:
        agent_neighbor = agent.neighbor(coord_all)
        agent.fuse(agent_neighbor, N, array_map)
    #agent移动
    for agent in list_agent:
        cell_consider = agent.cellconsider()
        agent.move(cell_consider)

    #循环计数
    count += 1
    print(count)

    #绘制热图
    map_im = float(1)/(np.exp(agent0.map) + 1)
    im.set_array(map_im)

    #绘制折线图
    coord_pl[0].append(agent0.coord[0])
    coord_pl[1].append(agent0.coord[1])
    if count > 100:
        coord_pl[0].pop(0)
        coord_pl[1].pop(0)
    ax.plot(coord_pl[0], coord_pl[1], color='red', marker='>', linewidth=0.3)
    ax.lines.pop(0)

    plt.pause(0.5)

    #循环退出条件
    unsure = 0#概率图的不确定度
    for agent in list_agent:
        for i in range(num_row):
            for j in range(num_column):
                unsure += np.exp(-2*np.linalg.norm(agent.map[i][j]))
    unsure_final = unsure / (N * num_column * num_row)
    if unsure_final <= 0.001:
        break
