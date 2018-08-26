#coding=utf-8
import numpy as np
from scipy import stats

class Agent:
    def __init__(self, index, coord, map, p, q, radius_search, radius_communicate, radius_consider, bound_speed,  bound_Q):
        '''
        :param index: agent编号，从0开始
        :param coord: agent坐标
        :param map: 单个agent概率图
        :param p: 探测准确率
        :param q: 错误报警率
        :param radius_search: 搜索半径
        :param radius_communicate: 通讯半径
        :param radius_consider: 移动时考虑的范围半径
        :param bound_speed: agent移动速度上限
        :param bound_Q: Q值上限
        '''
        self.index = index
        self.coord= coord
        self.map = map
        self.num_row = map.shape[0]
        self.num_volumn = map.shape[1]
        self.p = p
        self.q = q
        self.radius_search = radius_search
        self.radius_communicate = radius_communicate
        self.radius_consider = radius_consider
        self.bound_speed = bound_speed
        self.bound_Q = bound_Q

    def cellsearch(self):
        '''
        找出当前agent搜索范围内的所有cell。
        :return: cell坐标的列表
        '''
        cell_search = []
        array_agent = np.array(self.coord)
        # 遍历所有cell，挑出与当前agent距离小于search_radius的cell放入agent_cell。
        for i in range(self.num_row):
            for j in range(self.num_volumn):
                cell_coord = np.array([i, j])
                d = np.linalg.norm(array_agent - cell_coord)#计算每个cell与当前agent的距离。
                if d <= self.radius_search:
                    cell_search.append(cell_coord)
        return cell_search

    def cellconsider(self):
        '''
        找出当前agent移动时所需考虑的所有cell。
        :return: cell坐标的列表
        '''
        cell_consider = []
        array_agent = np.array(self.coord)
        # 遍历所有cell，挑出与当前agent距离小于search_radius的cell放入agent_cell。
        for i in range(self.num_row):
            for j in range(self.num_volumn):
                cell_coord = np.array([i, j])
                d = np.linalg.norm(array_agent - cell_coord)  # 计算每个cell与当前agent的距离。
                if d <= self.radius_consider:
                    cell_consider.append(cell_coord)
        return cell_consider

    def neighbor(self, coord_all):
        '''
        返回当前agent通讯范围内的agent编号
        :param coord_all:
        :return: agent编号
        '''
        agent_neighbor = []
        array_agent = np.array(self.coord)
        for coord in coord_all:
            array_otheragent = np.array(coord)
            d = np.linalg.norm(array_otheragent - array_agent)#计算此agent与当前agent的距离
            if d <= self.radius_communicate:
                agent_neighbor.append(1)
            else:
                agent_neighbor.append(0)
        return agent_neighbor

    def update(self, matrix_sr, cell_search):
        '''
        单个概率图更新。
        :param matrix_sr: 输入的搜索区域矩阵
        :param cell_search: 当前agent搜索范围内的cell
        :return: 返回单个agent更新后的概率图
        '''
        #遍历agent_cell，更新Q值。
        x = [1, 0]
        p = (self.p, 1 - self.p)
        q = (self.q, 1 - self.q)
        for cell in cell_search:
            exist = matrix_sr[cell[0]][cell[1]]

            #观测过程模拟。
            if exist:
                exist_ob = stats.rv_discrete(values=(x, p)).rvs(size=1)[0]
            else:
                exist_ob = stats.rv_discrete(values=(x, q)).rvs(size=1)[0]

            #根据观测结果更新Q值。
            if exist_ob:
                Q = self.map[cell[0]][cell[1]] + np.log(float(self.q)/float(self.p))
                Q_final = max(min(Q,self.bound_Q), -self.bound_Q)
                self.map[cell[0]][cell[1]] = Q_final
            else:
                Q = self.map[cell[0]][cell[1]] + np.log(float(1-self.q)/float(1-self.p))
                Q_final = max(min(Q, self.bound_Q), -self.bound_Q)
                self.map[cell[0]][cell[1]] = Q_final
        return self.map

    def fuse(self, agent_neighbor, N, array_map):
        '''
        agent与其通讯范围内的其他agent的概率图进行融合。
        :param agent_neighbor: 列表中，当前agent通讯范围内的agent（包括自己）对应的值为1，否则为0
        :param N: agent总数
        :param array_map: 由所有agent单独更新后的概率图组成的array
        :return: 返回融合之后的概率图
        '''
        d = sum(agent_neighbor)
        list_weight = [float(1)/N,] * N
        list_weight[self.index] = 1-(float(d-1)/N)
        array_weight = np.array(list_weight)
        array_weight_final = array_weight * np.array(agent_neighbor)
        map_fused = np.zeros(array_map[0].shape)
        #遍历所有agent的map和权值，相乘并累加。

        for weight, map in zip(array_weight_final, array_map):
            map_fused += weight * map
        self.map = map_fused
        return map_fused

    def move(self, cell_consider):
        '''
        根据概率分布移动agent。
        :param cell_consider: 当前agent需要考虑的cell
        :return: 移动后的agent坐标
        '''
        #计算质心坐标，决定移动速度。
        mass = 0
        centroid_top = np.array([0.0, 0.0])
        for cell in cell_consider:
            q = self.map[cell[0]][cell[1]]
            density = np.exp(-2 * np.linalg.norm(q))
            mass += density
            centroid_top += np.array(cell) * density
        centroid = centroid_top / mass
        speed = centroid - self.coord

        if np.linalg.norm(speed) > self.bound_speed:
            speed = (self.bound_speed / np.linalg.norm(speed)) * speed

        self.coord += speed
        return self.coord
