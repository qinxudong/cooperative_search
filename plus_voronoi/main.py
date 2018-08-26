#coding=utf-8
from agent import Agent
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def paint(Z):
    '''
    可视化输出概率图，三维图形。
    :param Z: z坐标矩阵
    :return: None
    '''
    figure = plt.figure()
    ax = Axes3D(figure)
    X = np.arange(0, 25)
    Y = np.arange(0, 25)

    X, Y = np.meshgrid(X, Y)
    plt.title("This is experiment")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(0,1)

    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')
    plt.show()

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
radius_communicate = 10

bound_speed = 1000
bound_Q = 10000000000000000

#agent初始化
agent0 = Agent(0, coords[0], map_init, p, q, radius_search, radius_communicate, bound_speed,  bound_Q)
agent1 = Agent(1, coords[1], map_init, p, q, radius_search, radius_communicate, bound_speed,  bound_Q)
agent2 = Agent(2, coords[2], map_init, p, q, radius_search, radius_communicate, bound_speed,  bound_Q)
agent3 = Agent(3, coords[3], map_init, p, q, radius_search, radius_communicate, bound_speed,  bound_Q)
agent4 = Agent(4, coords[4], map_init, p, q, radius_search, radius_communicate, bound_speed,  bound_Q)
agent5 = Agent(5, coords[5], map_init, p, q, radius_search, radius_communicate, bound_speed,  bound_Q)

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
    #概率图融合
    coord_all = []  # 所有agent坐标的列表
    list_map = []  # 所有agent概率图的列表
    for agent in list_agent:
        arr_coord = np.array(agent.coord)
        arr_coord.astype(np.float32)
        coord_all.append([arr_coord[0],arr_coord[1]])
        list_map.append(agent.map)

    array_map = np.array(list_map)  # 转array
    for agent in list_agent:
        agent_neighbor = agent.neighbor(coord_all)
        agent.fuse(agent_neighbor, N, array_map)
    #agent移动
    for agent in list_agent:
        cell_consider = agent.cell_consider(coord_all)
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
            #print(maps)
            paint(maps[0])

        '''
        #输出
        output = pd.DataFrame(maps[0])
        output.to_csv('output.csv', index=False)
        '''
        break
