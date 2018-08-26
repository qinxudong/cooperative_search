#coding=utf-8
from agent import Agent
import numpy as np
import pandas as pd
from util import draw_curve, draw_map
import matplotlib.pyplot as plt

class coverage_control_method(object):
    def __init__(self, size=(25, 25), num_target=8):
        self.size = size
        self.row = size[0]
        self.column = size[1]
        self.num_target = num_target

    def test(self, with_animation=False, with_map=False, with_curve=True, with_output=False):
        # 输入搜索区域矩阵数据
        # df = pd.read_csv('/home/qin/code/experiments/cooperative_search/input2.csv', header=None)
        # matrix_sr = df.values

        # 随机初始化目标位置
        matrix_sr = np.zeros((self.row, self.column))
        target_position_index = np.random.choice(self.row*self.column, self.num_target, replace=False)
        target_position_x = target_position_index // self.column
        target_position_y = target_position_index % self.column
        for x, y in zip(target_position_x, target_position_y):
            matrix_sr[x][y] = 1

        # 参数定义
        map_init = np.zeros([self.row, self.column])
        p = 0.9
        q = 0.3
        radius_search = 4
        radius_communicate = 25
        radius_consider = 6
        bound_speed = 2
        bound_Q = 10000000000000000

        # 随机初始化agent位置
        center_x = np.random.randint(radius_communicate//2, self.row - radius_communicate//2)
        center_y = np.random.randint(radius_communicate//2, self.column - radius_communicate//2)
        center = (center_x, center_y)
        coords_in = []
        for x in range(center[0] - radius_communicate//2, center[0] + radius_communicate//2 + 1):
            for y in range(center[1] - radius_communicate//2, center[1] + radius_communicate//2 + 1):
                if (pow(x-center[0], 2) + pow(y-center[1], 2)) <= pow(radius_communicate//2, 2):
                    coords_in.append((x, y))
        id_coords = list(np.random.choice(len(coords_in), size=6, replace=False))
        coords_in = np.array(coords_in)
        coords = coords_in[id_coords]
        coords = list(coords.astype(float))
        print(center, coords)


        # agent初始化
        agent0 = Agent(0, coords[0], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,
                       bound_Q)
        agent1 = Agent(1, coords[1], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,
                       bound_Q)
        agent2 = Agent(2, coords[2], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,
                       bound_Q)
        agent3 = Agent(3, coords[3], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,
                       bound_Q)
        agent4 = Agent(4, coords[4], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,
                       bound_Q)
        agent5 = Agent(5, coords[5], map_init, p, q, radius_search, radius_communicate, radius_consider, bound_speed,
                       bound_Q)

        # 初始化一些后面用到的列表
        list_agent = [agent0, agent1, agent2, agent3, agent4, agent5]  # agent列表
        N = len(list_agent)  # agent数量
        count = 0  # 循环次数
        unsures = []

        # 热图初始化
        if with_animation:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            map_im = float(1) / (np.exp(map_init) + 1)
            im = ax.imshow(map_im, cmap='bone_r', vmin=0, vmax=1)
            plt.colorbar(im, shrink=0.5)
            coord_pls = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
            colors = ['red', 'yellow', 'blue', 'c', 'green', 'm']
            for i in range(N):
                coord_pls[i][0].append(coords[i][0])
                coord_pls[i][1].append(coords[i][1])
            for i in range(N):
                ax.plot(coord_pls[i][1], coord_pls[i][0], color=colors[i], marker='>', linewidth=0.3)
            plt.pause(1)

        # 主循环
        while True:
            # 循环退出条件
            unsure = 0
            for agent in list_agent:
                for i in range(self.row):
                    for j in range(self.column):
                        unsure += np.exp(-2 * np.linalg.norm(agent.map[i][j]))
            unsure_final = unsure / (N * self.column * self.row)
            unsures.append(unsure_final)

            if unsure_final <= 0.001:
                if with_map:
                    for agent in list_agent:
                        maps = []
                        maps.append(float(1)/(np.exp(agent.map) + 1))
                        #print(maps)
                        draw_map(maps[0])
                if with_curve:
                    draw_curve(unsures)
                if with_output:
                    output = pd.DataFrame(maps[0])
                    output.to_csv('output.csv', index=False)
                break

            # 单个agent概率图更新
            for agent in list_agent:
                cell_search = agent.cellsearch()
                agent.update(matrix_sr, cell_search)
            # 概率图融合
            coord_all = []  # 所有agent坐标的列表
            list_map = []  # 所有agent概率图的列表
            for agent in list_agent:
                coord_all.append(agent.coord)
                list_map.append(agent.map)
            array_map = np.array(list_map)  # 转array
            for agent in list_agent:
                agent_neighbor = agent.neighbor(coord_all)
                agent.fuse(agent_neighbor, N, array_map)
            # agent移动
            for agent in list_agent:
                cell_consider = agent.cellconsider()
                agent.move(cell_consider)

            count += 1
            print(count)

            if with_animation:
                # 绘制热图
                map_im0 = np.zeros([self.row, self.column])
                for agent in list_agent:
                    map_im0 += float(1) / (np.exp(agent.map) + 1)
                map_im = map_im0 / N
                im.set_array(map_im)

                # 绘制折线图
                for i, agent in zip(range(N), list_agent):
                    coord_pls[i][0].append(agent.coord[0])
                    coord_pls[i][1].append(agent.coord[1])
                    if count >= 1:
                        coord_pls[i][0].pop(0)
                        coord_pls[i][1].pop(0)
                    ax.plot(coord_pls[i][1], coord_pls[i][0], color=colors[i], marker='>', linewidth=0.3)
                    ax.lines.pop(0)

                plt.pause(0.3)


agent = coverage_control_method()
agent.test(with_animation=True, with_map=False, with_curve=True, with_output=False)
print('change')





