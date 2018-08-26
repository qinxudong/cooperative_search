import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_curve(unsures):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(unsures)
    plt.show()


def draw_map(Z):
    '''
    可视化输出概率图，三维图形。
    :param Z: z坐标矩阵
    :return: None
    '''
    figure = plt.figure()
    ax = Axes3D(figure)
    X = np.arange(0, Z.shape[0])
    Y = np.arange(0, Z.shape[1])

    X, Y = np.meshgrid(X, Y)
    plt.title("This is an experiment.")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(0,1)

    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')
    plt.show()