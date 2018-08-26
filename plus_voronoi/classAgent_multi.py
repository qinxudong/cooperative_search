#coding=utf-8
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull
class Agent:
    def __init__(self, index, coord, map, p, q, radius_search, radius_communicate, Ku, bound_speed,  bound_Q):
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
        self.Ku = Ku
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

    def slope_opposite(self,point1, point2):
        k = float(point1[1] - point2[1]) / float(point1[0] - point2[0])

        return float(-1) / k

    def computeslope(self,point1, point2):
        k = float(point1[1] - point2[1]) / float(point1[0] - point2[0])

        return k

    def distance(self,point, line_list):

        dis_value = point[1] - line_list[2] - line_list[0] * (point[0] - line_list[1])
        return dis_value

    def imaginline(self,ridge_vertices, real_point):
        imaginline_list = []
        for item in enumerate(ridge_vertices):
            if item[1] == [-1, real_point]:
                imaginline_list.append(item[0])

        return imaginline_list

    def voronoi(self,coord_all):

        lines_list = []
        vor = Voronoi(coord_all)

        vertices = vor.vertices

        #voronoi_plot_2d(vor)

        new_coord = [self.coord[0], self.coord[1]]
        local = coord_all.index(new_coord)

        region_local = vor.point_region[local]

        vertices_list = vor.regions[region_local]
        #print vertices_list
        #print vor.vertices

        if len(vertices_list) == 2:
            if vertices_list[0] == -1 and vertices_list[1] != -1:
                imaginline_list = self.imaginline(vor.ridge_vertices, vertices_list[1])
                for num in imaginline_list:
                    point1 = vor.points[vor.ridge_points[num][0]]
                    point2 = vor.points[vor.ridge_points[num][1]]
                    lines_list.append([self.slope_opposite(point1, point2), vertices[vertices_list[1]][0],
                                       vertices[vertices_list[1]][1]])
            else:
                if vertices_list[0] != -1 and vertices_list[1] == -1:
                    imaginline_list = self.imaginline(vor.ridge_vertices, vertices_list[0])
                    for num in imaginline_list:
                        point1 = vor.points[vor.ridge_points[num][0]]
                        point2 = vor.points[vor.ridge_points[num][1]]
                        lines_list.append([self.slope_opposite(point1, point2), vertices[vertices_list[0]][0],
                                           vertices[vertices_list[0]][1]])


        else:
            for i in range(0, len(vertices_list) - 1):

                if vertices_list[i] != -1:
                    if vertices_list[i + 1] != -1:
                        k = self.computeslope(vertices[vertices_list[i]], vertices[vertices_list[i + 1]])
                        lines_list.append([k, vertices[vertices_list[i]][0], vertices[vertices_list[i]][1]])

                    else:

                        k1 = i
                        imaginline_list = self.imaginline(vor.ridge_vertices, vertices_list[k1])
                        if len(imaginline_list) == 1:

                            local_ridgevertices = vor.ridge_vertices.index([-1, vertices_list[k1]])

                            point1 = vor.points[vor.ridge_points[local_ridgevertices][0]]
                            point2 = vor.points[vor.ridge_points[local_ridgevertices][1]]
                            lines_list.append([self.slope_opposite(point1, point2), vertices[vertices_list[k1]][0],
                                               vertices[vertices_list[k1]][1]])

                        else:
                            for num in imaginline_list:
                                point1 = vor.points[vor.ridge_points[num][0]]
                                point2 = vor.points[vor.ridge_points[num][1]]

                                if np.all(point1 == self.coord) or np.all(point2 == self.coord):
                                    lines_list.append([self.slope_opposite(point1, point2), vertices[vertices_list[k1]][0],
                                                       vertices[vertices_list[k1]][1]])
                                    break

                        if i + 1 == len(vertices_list) - 1:
                            k1 = 0
                            local_ridgevertices = vor.ridge_vertices.index([-1, vertices_list[k1]])

                            point1 = vor.points[vor.ridge_points[local_ridgevertices][0]]
                            point2 = vor.points[vor.ridge_points[local_ridgevertices][1]]
                            lines_list.append([self.slope_opposite(point1, point2), vertices[vertices_list[k1]][0],
                                               vertices[vertices_list[k1]][1]])


                else:
                    if vertices_list[i + 1] != -1:

                        k2 = i + 1
                        imaginline_list = self.imaginline(vor.ridge_vertices, vertices_list[k2])
                        if len(imaginline_list) == 1:

                            local_ridgevertices2 = vor.ridge_vertices.index([-1, vertices_list[k2]])
                            point3 = vor.points[vor.ridge_points[local_ridgevertices2][0]]
                            point4 = vor.points[vor.ridge_points[local_ridgevertices2][1]]
                            lines_list.append([self.slope_opposite(point3, point4), vertices[vertices_list[k2]][0],
                                               vertices[vertices_list[k2]][1]])


                        else:
                            for num in imaginline_list:
                                point1 = vor.points[vor.ridge_points[num][0]]
                                point2 = vor.points[vor.ridge_points[num][1]]

                                if np.all(point1 == self.coord) or np.all(point2 == self.coord):
                                    lines_list.append([self.slope_opposite(point1, point2), vertices[vertices_list[k2]][0],
                                                       vertices[vertices_list[k2]][1]])
                                    break
                        if i == 0:
                            k2 = len(vertices_list) - 1

                            local_ridgevertices2 = vor.ridge_vertices.index([-1, vertices_list[k2]])
                            point3 = vor.points[vor.ridge_points[local_ridgevertices2][0]]
                            point4 = vor.points[vor.ridge_points[local_ridgevertices2][1]]
                            lines_list.append([self.slope_opposite(point3, point4), vertices[vertices_list[k2]][0],
                                               vertices[vertices_list[k2]][1]])


            if vertices_list[0] != -1 and vertices_list[len(vertices_list) - 1] != -1:

                k = self.computeslope(vertices[vertices_list[0]], vertices[vertices_list[len(vertices_list) - 1]])
                lines_list.append([k, vertices[vertices_list[0]][0], vertices[vertices_list[0]][1]])


        #print lines_list
        return lines_list

    def judge(self,number):
        if number > 0:
            label = 1
        else:
            label = 0
        return label
    def cell_consider(self,coord_all):
        inside_list = []
        dis_coordlist = []
        lines_list = self.voronoi(coord_all)

        for one in lines_list:
            dis_coord = self.distance(self.coord, one)
            dis_coordlist.append(self.judge(dis_coord))
        #print dis_coordlist
        for i in range(self.num_row):
            for j in range(self.num_volumn):
                point = [i, j]
                #print point
                k = 0
                Flag = True
                for one in lines_list:
                    dis_point = self.distance(point, one)
                    # print point

                    if self.judge(dis_point) != dis_coordlist[k]:
                        # print judge(dis_point)
                        Flag = False
                        break
                    else:
                        k += 1
                while Flag:
                    inside_list.append(point)
                    break


        '''plt.xlim(xmax=25, xmin=0)
        plt.ylim(ymax=25, ymin=0)

        x = []
        y = []
        for point in inside_list:
            x.append(point[0])
            y.append(point[1])

        plt.plot(x, y, 'ro', color='red', markersize=1)

        plt.legend(loc='upper center', shadow=True, fontsize='x-large')
        plt.grid(True)

        for i in lines_list:
            y1 = i[0] * (x - i[1]) + i[2]

            plt.plot(x, y1)

        plt.pause(0.5)'''
        return inside_list
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
        map_fused = np.zeros([25, 25])
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
            q = self.map[int(cell[0])][int(cell[1])]
            density = np.exp(-2 * np.linalg.norm(q))
            mass += density
            centroid_top += np.array(cell) * density
        centroid = centroid_top / mass
        speed = self.Ku*(centroid - self.coord)
        print(np.linalg.norm(speed))

        if np.linalg.norm(speed) > self.bound_speed:
            speed = (self.bound_speed / np.linalg.norm(speed)) * speed

        self.coord += speed
        return self.coord
