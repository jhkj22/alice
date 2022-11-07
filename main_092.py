import glob
import itertools
import sys
import typing
from enum import Enum

import numpy as np
import cv2

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow,
                             QVBoxLayout, QHBoxLayout, QPushButton, QTreeView, QAbstractItemView)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg as qt5agg
import matplotlib.figure


def upload_figure_to_google_drive(figure):
    sys.path.append('../')
    from google_drive_api.upload_file import upload_png # type: ignore

    import tempfile

    dirpath = tempfile.mkdtemp()

    path = dirpath + '/ml.png'
    figure.savefig(path)
    upload_png(path)

    import shutil
    shutil.rmtree(dirpath)

    print('upload done', flush=True)


class ContsPlotWindow(QMainWindow):
    def __init__(self, grid=(1, 2)):
        QMainWindow.__init__(self)

        self.setGeometry(0, 0, 1920, 600)

        self.frame = QWidget(self)
        self.setCentralWidget(self.frame)

        self.frame.setLayout(QHBoxLayout())
        self.frame.layout().setSpacing(0)
        self.frame.layout().setContentsMargins(0, 0, 0, 0)

        self.init_tree_view()

        self.grid = grid
        self.init_figure()

        self.l_plotter = {}

    def init_tree_view(self):
        self.tree = QTreeView(self.frame)
        self.tree.setMaximumWidth(200)

        self.frame.layout().addWidget(self.tree)

        self.model = QStandardItemModel(0, 1)
        self.model.setHeaderData(0, Qt.Horizontal, 'object')
        self.tree.setModel(self.model)

        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.tree.selectionModel().selectionChanged.connect(
            self.on_tree_selection_changed
        )

    def __get_key(self, index):
        indices = []
        while index.isValid():
            indices.insert(0, index.row())
            index = index.parent()
        return ','.join([str(i) for i in indices])

    def on_tree_selection_changed(self, sel):
        for child in ax[1].get_children():
            if type(child) is matplotlib.quiver.Quiver:
                child.remove()
            if type(child) is matplotlib.collections.PathCollection:
                child.remove()
            if type(child) is matplotlib.patches.Arc:
                child.remove()
            if type(child) is matplotlib.collections.LineCollection:
                child.remove()

        for line in ax[1].get_lines():
            line.remove()

        for index in self.tree.selectedIndexes():
            k = self.__get_key(index)

            if k not in self.l_plotter:
                continue

            plotter = self.l_plotter[k]

            plotter.plot(ax[1])

        self.canvas.draw()

    def init_figure(self):
        figure = QWidget(self.frame)
        self.frame.layout().addWidget(figure)

        figure.setLayout(QVBoxLayout())
        figure.layout().setSpacing(0)
        figure.layout().setContentsMargins(0, 0, 0, 0)

        self.canvas = qt5agg.FigureCanvasQTAgg(matplotlib.figure.Figure())
        toolbar = qt5agg.NavigationToolbar2QT(self.canvas, figure)

        button = QPushButton('Upload')
        button.clicked.connect(
            lambda: upload_figure_to_google_drive(self.canvas.figure))

        toolbar.addWidget(button)
        figure.layout().addWidget(toolbar)
        figure.layout().addWidget(self.canvas)

        self.ax = []

        grid = self.grid
        n = grid[0] * 100 + grid[1] * 10
        for i in range(grid[0] * grid[1]):
            ax = self.canvas.figure.add_subplot(n + i + 1)
            self.ax.append(ax)

    def __add_plotter(self, item, plotter):
        key = self.__get_key(item.index())
        self.l_plotter[key] = plotter

    def append_cont(self, tree_cont, tag='group'):
        def _depth():
            item = tree_cont

            # default depth is 0 as cont = [[0, 0], [0, 0] ... ]
            depth = -2

            while True:
                if isinstance(item, list):
                    depth += 1
                    if len(item) == 0:
                        # 0番目がたまたま空配列の場合の対応
                        # cont = [[], [[0, 0], [1, 1]]]
                        depth += 1
                        break
                    item = item[0]
                else:
                    break

            return depth

        depth = _depth()

        if depth > 2:
            return

        group = QStandardItem(tag)
        self.model.appendRow(group)

        if depth == 0:
            cont = tree_cont
            self.__add_plotter(group, Plotter2([cont]))

        if depth == 1:
            conts = tree_cont
            self.__add_plotter(group, Plotter2(conts))

            for i, cont in enumerate(conts):
                empty = ' (empty)' if not cont else ''
                item1 = QStandardItem('cont - ' + str(i) + empty)
                group.appendRow(item1)

                self.__add_plotter(item1, Plotter2([cont]))

        class PlotterLConts:
            def __init__(self, l_conts):
                self.l_conts = l_conts

            def plot(self, ax):
                for conts in self.l_conts:
                    plot_conts2(ax, conts)

        if depth == 2:
            l_conts = tree_cont
            self.__add_plotter(group, PlotterLConts(l_conts))

            for i, conts in enumerate(l_conts):
                item1 = QStandardItem('conts - ' + str(i))
                group.appendRow(item1)

                self.__add_plotter(item1, Plotter2(conts))

                for j, cont in enumerate(conts):
                    empty = ' (empty)' if not cont else ''
                    item2 = QStandardItem('cont - ' + str(j) + empty)
                    item1.appendRow(item2)

                    self.__add_plotter(item2, Plotter2([cont]))

    def append_map_conts(self, image_map, tag='group'):
        group = QStandardItem(tag)
        self.model.appendRow(group)

        conts_ext = []
        for r, c, conts in image_map.get_list():
            conts_ext.extend(conts)

        plotter = Plotter2(conts_ext)
        self.__add_plotter(group, plotter)

        for r, c, conts in image_map.get_list():
            item = QStandardItem('conts {:},{:}'.format(r, c))
            group.appendRow(item)

            plotter = Plotter2(conts)
            self.__add_plotter(item, plotter)

    def append_plot(self, plotter, tag='group'):
        group = QStandardItem(tag)
        self.model.appendRow(group)

        self.__add_plotter(group, plotter)

    def append_plots(self, plotters, tag='group'):
        group = QStandardItem(tag)
        self.model.appendRow(group)

        for i, plotter in enumerate(plotters):
            item = QStandardItem(tag + ' - ' + str(i))
            group.appendRow(item)
            self.__add_plotter(item, plotter)


app = QApplication(sys.argv)

conts_tree_view = ContsPlotWindow()

ax = conts_tree_view.ax

# angle is degree


def rotate_and_clip(img, r1, r2, c1, c2, angle):
    cx, cy = (c1 + c2) / 2, (r1 + r2) / 2
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1)

    img = cv2.warpAffine(img, m, img.shape[:2])
    img = img[r1: r2, c1: c2]

    return img


def load_cat():
    img = cv2.imread('../res/cat.jpg')
    for _ in range(4):
        img = cv2.pyrDown(img)

    return img


def load_dead_tree():
    img = cv2.imread('../res/dead_tree.jpg')
    for _ in range(4):
        img = cv2.pyrDown(img)

    return img


def load_dog2():
    img = cv2.imread('../res/dog2.jpg')
    for _ in range(3):
        img = cv2.pyrDown(img)

    return img


def load_french_toast():
    img = cv2.imread('../res/french-toast.jpg')
    for _ in range(4):
        img = cv2.pyrDown(img)

    return img[7: 78, 0: 69]


def load_koara():
    img = cv2.imread('../res/koara.jpg')
    for _ in range(4):
        img = cv2.pyrDown(img)

    return img


def load_oranghutan():
    img = cv2.imread('../res/oranghutan.jpg')
    for _ in range(3):
        img = cv2.pyrDown(img)

    return img


def load_tiger1():
    img = cv2.imread('../res/tiger-1.jpeg')
    for _ in range(5):
        img = cv2.pyrDown(img)

    return img[0: 112, 46: 163]


def load_tiger2():
    img = cv2.imread('../res/tiger-2.jpeg')
    for _ in range(6):
        img = cv2.pyrDown(img)

    return img


def load_tiger3():
    img = cv2.imread('../res/tiger-3.jpeg')
    for _ in range(5):
        img = cv2.pyrDown(img)

    return img

#img = load_french_toast()
#img = load_dog2()
#img = load_dead_tree()
#img = load_cat()
#img = load_koara()
#img = load_oranghutan()
#img = load_tiger3()


base = '../res/hand4/rock'

l = sorted(glob.glob(base + '/*.jpg'))
im_index = 299
img = cv2.imread(l[im_index])

# blob-connect: 118, 119, 120, 122
# 手のひら影: 299, 300, 310, 311, 312,

# img = cv2.resize(img, (120, 120))
# img = cv2.pyrDown(img)

if im_index == 118:
    img = img[13: 55, 8: 48][4: 26, 9: 33]
if im_index == 119:
    img = img[10: 42, 2: 34]
if im_index == 180:
    img = img[9: 45, 14: 47]
if im_index == 299:
    img = img[30: 107, 0: 81]
    img = img[21: 48, 0: 28] # 人差し指
    # img = img[30: 54, 7: 34] # 親指
    # img = img[35: 57, 33: 59] # 薬指
    # img = img[32: 56, 51: 71] # 小指
    # img = img[60: 74, 5: 20]
    # img = img[9: 22, 5: 20]
    # img = img[0: 14, 16: 32]
    # img = img[63: 78, 27: 43]
# if im_index == 311:
#     img = img[12: 35, 36: 56]

img = np.flip(img, axis=2)
img_chw = np.transpose(img, (2, 0, 1))
img_chw = img_chw.astype(np.float)
_, rr, cc = img_chw.shape
img_r, img_g, img_b = img_chw

img_only = True
img_only = False


def log(*args):
    f_name = sys._getframe(1).f_code.co_name
    print('[' + f_name + ']', *args)


def plot_conts(ax, conts, r=0, c=0, marker='-', ms=3, cl='r', lw=1):
    ax.add_collection(matplotlib.collections.LineCollection(
        [[[p[0] + c, p[1] + r] for p in cont] for cont in conts if len(cont)],
        color=cl,
        lw=lw
    ))
    # for cont in conts:
    #     if len(cont) == 0:
    #         continue
    #     l = np.transpose(cont)
    #     ax.plot(l[0] + c, l[1] + r, marker, ms=ms, color=cl, lw=lw)


def plot_conts2(ax, conts):
    return plot_conts(ax, conts, 0.5, 0.5, '-', cl=(1, 0, 0, 0.5))


def plot_rect(ax, r, c, rs, cs):
    return ax.add_patch(plt.Rectangle(xy=[c - 0.5, r - 0.5],
                                      width=cs, height=rs, fill=False, ec='b'))


class Plotter:
    def __init__(self, conts=[], r=0, c=0, marker='-', ms='3', cl='r', lw=1):
        self.conts = conts
        self.r = r
        self.c = c
        self.marker = marker
        self.ms = ms
        self.cl = cl
        self.lw = lw

    def plot(self, ax):
        return plot_conts(ax, self.conts, r=self.r, c=self.c, marker=self.marker, ms=self.ms, cl=self.cl, lw=self.lw)


class Plotter2:
    def __init__(self, conts=[]):
        self.conts = conts

    def plot(self, ax):
        return plot_conts2(ax, self.conts)


class PlotterPoints:
    def __init__(self, points, cl='r', ms=3):
        self.points = points
        self.cl = cl
        self.ms = ms

    def plot(self, ax):
        if len(self.points) == 0:
            return None

        l = np.transpose(self.points)
        return ax.plot(l[1], l[0], 'o', color=self.cl, ms=self.ms)


class PlotterQuiver:
    def __init__(self, x, y, dx, dy, p, width=0.0075, head_w=3, head_l=5):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.p = p
        self.width = width
        self.head_w = head_w
        self.head_l = head_l

    def plot(self, ax):
        return ax.quiver(self.x, self.y, self.dx, self.dy, self.p, scale_units='xy', scale=1, width=self.width, headwidth=self.head_w, headlength=self.head_l)


class PlotterScatter:
    def __init__(self, x, y, c, s):
        self.x = x
        self.y = y
        self.c = c
        self.s = s

    def plot(self, ax):
        return ax.scatter(self.x, self.y, c=self.c, s=self.s)


class PlotterArcs:
    def __init__(self, l_xy, l_angle, l_theta1, l_theta2):
        self.l_xy = l_xy
        self.l_angle = l_angle
        self.l_theta1 = l_theta1
        self.l_theta2 = l_theta2

    def plot(self, ax):
        for xy, angle, theta1, theta2 in zip(self.l_xy, self.l_angle, self.l_theta1, self.l_theta2):
            arc = matplotlib.patches.Arc(
                xy, 0.2, 0.4, angle=angle, theta1=theta1 - angle, theta2=theta2 - angle,
                lw=0.6, ec='green')
            ax.add_patch(arc)


def box_in(r, c, rb=rr, cb=cc):
    return r >= 0 and c >= 0 and r < rb and c < cb


def indices_nC2(n):
    l = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            l.append([i, j])
    return l


def range_adj(length):
    l = list(range(length))
    return zip(l[:-1], l[1:])


def chaincode_dif_abs(j1, j2):
    dj = abs(j1 - j2)
    dj = min([dj, 8 - dj])
    return dj


def chaincode_dif_clock(j1, j2):
    return (j2 - j1) % 8


def chaincode_mid(j1, j2):
    m = (j1 + j2) / 2
    if abs(j1 - j2) > 4:
        m = (m + 4) % 8
    return m

# 0 <= r <= 1


def calc_ratio(v1, v2):
    v1, v2 = (v1, v2) if v1 < v2 else (v2, v1)
    return v1 / v2 if v2 > 0 else 0


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def calc_angle(r0, c0, r1, c1):
    dr, dc = r1 - r0, c1 - c0
    dx, dy = dc, -dr
    return np.arctan2(dy, dx)

# contの順にp1, p2, p3
# 直線が0, 正の曲率は0-180度, 負は逆


def interior_angle(p1, p2, p3):
    p = np.array([p1[0] + p1[1] * 1j,
                  p2[0] + p2[1] * 1j,
                  p3[0] + p3[1] * 1j])
    p -= p[1]
    p[0] *= -1
    if p[0] == 0:
        return 0
    p /= p[0]
    return np.angle(p)[2]

# Ray casting algorithm


def inside_polygon(p0, qs):
    cnt = 0
    L = len(qs)
    x, y = p0
    for i in range(L):
        x0, y0 = qs[i-1]
        x1, y1 = qs[i]
        x0 -= x
        y0 -= y
        x1 -= x
        y1 -= y

        cv = x0*x1 + y0*y1
        sv = x0*y1 - x1*y0
        if sv == 0 and cv <= 0:
            # a point is on a segment
            return True

        if not y0 < y1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        if y0 <= 0 < y1 and x0*(y1 - y0) > y0*(x1 - x0):
            cnt += 1
    return (cnt % 2 == 1)


def linear_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1

    return a, b, c


def distance_from_line(ps, a, b, c):
    x, y = np.transpose(ps)
    ld = (a * x + b * y + c) / (a ** 2 + b ** 2) ** 0.5
    return ld


def perpendicular_to_line(ps, a, b, c):
    x, y = np.transpose(ps)

    t = -(a * x + b * y + c) / (a ** 2 + b ** 2)

    x = x + t * a
    y = y + t * b

    return np.transpose([x, y])


def bisector(a1, b1, c1, a2, b2, c2):
    A1 = (a1 ** 2 + b1 ** 2) ** 0.5
    A2 = (a2 ** 2 + b2 ** 2) ** 0.5

    eq1 = np.array([a1, b1, c1])
    eq2 = np.array([a2, b2, c2])

    l_eq = [eq1 * A2 + sign * eq2 * A1 for sign in [1, -1]]
    l_eq = [[a, b, c] for a, b, c in l_eq if a != 0 or b != 0]

    return l_eq


def fit_circle(ps):
    N = len(ps)
    x, y = np.transpose(ps)

    M1 = np.array([
        [
            np.sum(x ** 2),
            np.sum(x * y),
            np.sum(x),
        ],
        [
            np.sum(x * y),
            np.sum(y ** 2),
            np.sum(y)
        ],
        [
            np.sum(x),
            np.sum(y),
            N
        ],
    ])

    M2 = np.array([
        -np.sum(x ** 3 + x * y ** 2),
        -np.sum(x ** 2 * y + y ** 3),
        -np.sum(x ** 2 + y ** 2),
    ])

    A, B, C = np.linalg.inv(M1) @ M2

    a = -A / 2
    b = -B / 2
    r = (a ** 2 + b ** 2 - C) ** 0.5

    return a, b, r


def bresenham(start, end):
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    d = 2 * dy - dx
    y = 0

    points = []
    for x in range(dx + 1):
        points.append([x0 + x * xx + y * yx, y0 + x * xy + y * yy])
        if d >= 0:
            y += 1
            d -= 2 * dx
        d += 2 * dy
    return points


def connect_line(line, tolerance=1):
    l_result = [[-99999]]
    for i in line:
        i2 = l_result[-1][-1]
        if i - i2 <= tolerance:
            l_result[-1].append(i)
        else:
            l_result.append([i])
    return l_result[1:]

def connect_line_bb(line, tolerance=1):
    l_result = [[-99999]]
    for i in range(len(line)):
        if line[i] == 0:
            continue
        i2 = l_result[-1][-1]
        if i - i2 <= tolerance:
            l_result[-1].append(i)
        else:
            l_result.append([i])
    return l_result[1:]


def connect_line_bb_circular(line):
    l = connect_line_bb(line)
    if len(l) > 1:
        if l[0][0] == 0 and l[-1][-1] == len(line) - 1:
            l[0] = l[-1] + l[0]
            l.pop(-1)
    return l


def pow_sobel(im):
    im = im.astype(np.uint8)
    sobel_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    sobel_x /= 4
    sobel_y /= 4
    return (sobel_x ** 2 + sobel_y ** 2) ** 0.5


def angle_between_vecs(v1, v2):
    l1, l2 = [np.linalg.norm(v) for v in [v1, v2]]
    if l1 == 0 or l2 == 0:
        return 0
    cos = np.dot(v1, v2) / (l1 * l2)
    cos = min([1, max([-1, cos])])
    return np.arccos(cos)


def mod_radian_0_2pi(t):
    return np.mod(t, 2 * np.pi)


def angle_between_tans(t1, t2):
    t1, t2 = [mod_radian_0_2pi(t) for t in [t1, t2]]
    angle = np.abs(t1 - t2)
    angle_rev = 2 * np.pi - angle
    angle = np.where(angle < angle_rev, angle, angle_rev)
    return angle


def angle_of_rotation(t0, t1):
    t0, t1 = [mod_radian_0_2pi(t) for t in [t0, t1]]
    dt = t1 - t0
    dt = np.where(dt < -np.pi, dt + 2 * np.pi, dt)
    dt = np.where(dt >= np.pi, dt - 2 * np.pi, dt)
    return dt


def grad_filter_2x2(img):
    # Roberts cross
    k1 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    k2 = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])

    img_x = cv2.filter2D(img, -1, k1)
    img_y = cv2.filter2D(img, -1, k2)
    img_x = img_x[:-1, :-1]
    img_y = img_y[:-1, :-1]

    return img_x, img_y


def get_grad(img):
    img_x, img_y = grad_filter_2x2(img)

    img_p = (img_x ** 2 + img_y ** 2) ** 0.5
    img_t = np.arctan2(img_y, img_x) + np.pi / 4

    img_x = np.cos(img_t)
    img_y = np.sin(img_t)
    img_t = np.arctan2(img_y, img_x)

    return [img_x, img_y, img_p, img_t]


img_x, img_y, img_p, img_t = get_grad(img_b)
y, x = np.where(img_t < np.infty)

plotter = PlotterQuiver(x + 0.5, y + 0.5, img_x * 0.7, img_y * 0.7, img_p)
conts_tree_view.append_plot(plotter, 'roberts')


def grad_cont(cont):
    l = []
    for c, r in cont:
        if not is_in_box(r, c):
            continue
        l.append(img_p[r, c])
    return l


def shape_descriptor(cont):
    n = len(cont) // 2 - 2
    n = max([1, n])
    la = []
    for i in range(n, len(cont) - n):
        a = interior_angle(cont[i - n], cont[i], cont[i + n]) / np.pi
        la.append(a)
    la = np.array(la)

    # [line|(half circle)|circle, normal|rev]
    if np.all(la > 0):
        if np.sum(la) >= 1.0:
            return [2, 0]
        elif np.sum(la) >= 0.5:
            return [1, 0]
    elif np.all(la < 0):
        if np.sum(la) <= -1.0:
            return [2, 1]
        elif np.sum(la) <= -0.5:
            return [1, 1]
    return [0, 0]


def optimize_sut_l(cont1, cont2, th_k):
    M = np.zeros((len(cont1), len(cont2)))
    for i in range(len(cont1)):
        for j in range(len(cont2)):
            k = np.linalg.norm(cont1[i] - cont2[j])
            M[i, j] = k

    T = np.zeros_like(M)
    for i in range(len(cont1)):
        for j in range(len(cont2)):
            if M[i, j] > th_k:
                continue

            min_i = np.min(M[i])
            min_j = np.min(M[:, j])

            if M[i, j] == min_i and M[i, j] == min_j:
                T[i, j] = 1
    # print(M); print(T)
    l = np.where(T == 1)
    return np.transpose(l)


def suture_line(cont1, cont2, th_k):
    _rev = False
    if len(cont1) < len(cont2):
        cont1, cont2 = cont2, cont1
        _rev = True

    sd1 = shape_descriptor(cont1)
    sd2 = shape_descriptor(cont2)
    if sd1[0] > 0 and sd2[0] > 0:
        if sd1[1] != sd2[1]:
            pass
            # return []
    if sd1[0] == 0 or sd2[0] == 0:
        if len(cont2) / len(cont1) < 0.7:
            pass
            # return []

    l = optimize_sut_l(cont1, cont2, th_k)

    if _rev:
        l = np.flip(l, axis=1)
        return l[np.argsort(l[:, 0])]  # cont1でソート
    else:
        return np.array(l)


def suture_line_i(cont1, cont2, th_k=1.5):
    cont1 = np.array(cont1)
    cont2 = np.array(cont2)
    l_sli = suture_line(cont1, cont2, th_k)

    l_slp = []
    l_d = []
    for i1, i2 in l_sli:
        p1 = cont1[i1]
        p2 = cont2[i2]
        l_d.append(np.linalg.norm(p1 - p2))
        l_slp.append([p1, p2])
    l_slp = np.array(l_slp)
    l_d = np.array(l_d)
    return [l_sli, l_slp, l_d]


def pop_next_candidate(l_rc_bg, img_done):
    while len(l_rc_bg) > 0:
        r, c = l_rc_bg.pop(0)
        if img_done[r, c] == 0:
            return [r, c]

    return []


def mark_img(cont, img):
    for c, r in cont:
        if c < 0 or r < 0:
            continue
        img[r, c] = 1


def clip_in_box(img, r, c, N):
    r1 = max([0, r - N])
    r2 = min([rr, r + N + 1])
    c1 = max([0, c - N])
    c2 = min([cc, c + N + 1])
    return img[r1: r2, c1: c2]


def get_rc(r1, r2, c1, c2):
    l = []
    for r in range(r1, r2):
        for c in range(c1, c2):
            l.append([r, c])
    return l


def is_in_box(r, c):
    if r < 0 or c < 0 or r >= rr - 1 or c >= cc - 1:
        return False
    return True


def norm_cont(cont):
    cont = np.array(cont)
    d = cont[0] - cont[-1]
    d = np.array([d[0], -d[1]])
    if np.linalg.norm(d) == 0:
        return []
    d = d / np.linalg.norm(d)
    return d


def tan_cont(cont):
    cont = np.array(cont)
    d = cont[0] - cont[-1]
    d = np.array([d[1], d[0]])
    if np.linalg.norm(d) == 0:
        return []
    d = d / np.linalg.norm(d)
    return d


def get_8d(r0=0, c0=0):
    l = [[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
    return [[r0 + ro, c0 + co] for ro, co in l]


def quantize_grad_t(t, n_neighbor=8):
    step_i = 1 if n_neighbor == 8 else 2

    li = range(0, 8, step_i)
    lt = [i / 4 * np.pi for i in li]
    l_dt = angle_between_tans(lt, t)

    lj = np.argsort(l_dt)
    l = [[li[j], l_dt[j] / np.pi] for j in lj]

    return l


def circular_dif(l, mn, mx):
    l_dif = []
    for i in range(len(l) - 1):
        l_dif.append(l[i + 1] - l[i])
    l_dif.append((mx - l[-1]) + (l[0] - mn))
    return l_dif


def min_interval_of_ps_circular(l, mn, mx):
    l = sorted(l)
    return mx - mn - max(circular_dif(l, mn, mx))


def min_range_of_ps_circular(l, mn, mx):
    l = sorted(l)
    l_dif = circular_dif(l, mn, mx)
    i = np.argmax(l_dif)
    return l[(i + 1) % len(l)], l[i]


def mean_polar(lp, lt):
    lp = np.array(lp)
    lx = lp * np.cos(lt)
    ly = lp * np.sin(lt)
    x, y = np.mean(lx), np.mean(ly)

    p = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(y, x)
    return p, t


def big_grad():
    _, _, img_p, _ = get_grad(img_b)
    # ax[1].imshow(img_p)
    l = np.argsort(img_p.flatten())[::-1]
    l2 = []
    for i in l:
        r = i // (cc - 1)
        c = i % (cc - 1)
        if r < 1 or c < 1 or r >= rr - 2 or c >= cc - 2:
            continue
        l2.append([r, c])
    # print(l2)
    return l2


T = typing.TypeVar('T')


class ImageMap(typing.Generic[T]):
    def __init__(self):
        self.dict = dict()

    def append(self, r: float, c: float, o: T):
        self.dict[ImageMap.__key(r, c)] = o

    def remove(self, r: float, c: float):
        self.dict.pop(ImageMap.__key(r, c))

    def get(self, r: float, c: float) -> typing.Union[T, None]:
        return self.dict.get(ImageMap.__key(r, c))

    def get_list(self):
        l = []
        for key in self.dict:
            r, c = [ImageMap.__number(v) for v in key.split(',')]
            o = self.dict[key]
            l.append([r, c, o])
        return sorted(l, key=lambda o: (o[0], o[1]))

    def __key(r, c):
        return ','.join([str(ImageMap.__number(v)) for v in [r, c]])

    def __number(v):
        fl = float(v)
        integer = int(fl)
        return integer if fl == integer else fl


class ExtendableCont:
    def __init__(self):
        self.cont = []
        self.id_min = 0

    def append(self, p):
        self.cont.append(p)

    def prepend(self, p):
        self.cont.insert(0, p)
        self.id_min -= 1

    def index(self, id):
        return id - self.id_min

    def id(self, index):
        return self.id_min + index

    def at(self, id):
        return self.cont[self.index(id)]

    def slice(self, id_begin, id_end):
        return self.cont[self.index(id_begin): self.index(id_end)]

class Cut2x2:

    INVALID = -1
    NONPOLAR = 0
    NORMAL = 1

    def __init__(self, im):
        self.im = np.array(im)
        self.i = self._index()
        self.type = self._type()
        self.di = self._di()
        self.ps = self._ps()

    def _index(self):
        l = self.im.ravel()
        return sum([b * 2 ** i for i, b in enumerate(l)])

    def _type(self):
        if min([self.i, 0xf - self.i]) == 0x0:
            return self.INVALID

        if min([self.i, 0xf - self.i]) == 0x6:
            return self.NONPOLAR

        return self.NORMAL

    LIST_DI = [
        [], [4, 2], [2, 0], [4, 0], [6, 4], [6, 2], [], [6, 0],
        [0, 6], [], [2, 6], [4, 6], [0, 4], [0, 2], [2, 4], [],
    ]

    def _di(self):
        return self.LIST_DI[self.i]

    LIST_P = [[0.5, 1], [0, 0.5], [0.5, 0], [1, 0.5]]

    @classmethod
    def _p2pi(cls, p):
        return cls.LIST_P.index(p) * 2

    @classmethod
    def _pi2p(cls, pi):
        return cls.LIST_P[pi // 2]

    def _ps(self):
        return [self._pi2p(i) for i in self.di]

    @classmethod
    def new_from_ps(cls, ps):
        di = [cls._p2pi(p) for p in ps]
        i = cls.LIST_DI.index(di)

        l = format(i, '04b')[::-1]
        l = np.array([int(d) for d in l])

        im = l.reshape(2, 2)

        return cls(im)

    def _grad_i(self):
        ps = np.array(self.ps)

        dp = ps[0] - ps[1]
        if abs(dp[0]) == 0.5:
            dp *= 2

        dr, dc = [int(o) for o in dp]

        return (get_8d().index([dr, dc]) + 6) % 8


def img_dif_2x2():
    img_dif = np.zeros_like(img_p)

    for r, c in get_rc(0, rr - 1, 0, cc - 1):
        im = img_b[r: r + 2, c: c + 2]
        img_dif[r, c] = np.max(im) - np.min(im)

    return img_dif


img_dif = img_dif_2x2()


def plot_edge_2x2():
    def _pat_grad(im):
        dv = np.ravel(im[1:] - im[:-1])
        dh = np.ravel(im[:, 1:] - im[:, :-1])
        return dv[0] * dv[1] > 0 and dh[0] * dh[1] > 0

    from typing import List

    def _pat_grad_diag(im, t):
        if not _pat_grad(im):
            return False

        grad_i, eps = quantize_grad_t(t)[0]
        if grad_i % 2 == 0:
            return False

        return eps < 1 / 14

    def _pat_bin(im, t) -> List[Cut2x2]:
        def _across_edge(im_th):
            l_rc = [
                [[0, 0], [0, 1]],
                [[1, 0], [1, 1]],
                [[0, 0], [1, 0]],
                [[0, 1], [1, 1]]
            ]
            for (r1, c1), (r2, c2) in l_rc:
                if im_th[r1, c1] == im_th[r2, c2]:
                    continue
                d = abs(im[r1, c1] - im[r2, c2])
                if d <= 2:
                    return False
            return True

        # /|
        # 5 | 2
        # 3 | 2
        # => 5 - 3 > 3 - 2
        def _remove_edge_indide_edge(cuts: 'list[Cut2x2]'):
            def _difs(cut: Cut2x2):
                def _dif(r, c):
                    if isinstance(r, int):
                        return im[r, 0] - im[r, 1]
                    if isinstance(c, int):
                        return im[0, c] - im[1, c]
                return [abs(_dif(*p)) for p in cut.ps]

            cut0 = None
            ds0 = None
            l_ds = []

            for cut in cuts:
                ds = _difs(cut)
                if cut._grad_i() % 2 == 0:
                    cut0 = cut
                    ds0 = ds
                else:
                    l_ds.append(ds)

            if ds0 is None:
                return cuts

            # print(ds0)
            # print(l_ds)

            for ds1 in l_ds:
                if ds0[0] == ds1[0]:
                    dds = [ds0[1], ds1[1]]
                elif ds0[1] == ds1[1]:
                    dds = [ds0[0], ds1[0]]

                if dds[0] < dds[1]:
                    cuts.remove(cut0)
                    break

            return cuts

        if _pat_grad_diag(im, t):
            mx, mn = np.max(im), np.min(im)
            l_th = [mn + 1, mx - 1]
        else:
            l = np.sort(im.ravel())
            ld = l[1:] - l[:-1]

            li = np.where(ld > 0)[0]
            l_th = [(l[i] + l[i + 1]) / 2 for i in li]

        cuts = []
        check_pattern = 0

        for th in l_th:
            im_th = np.where(im > th, 1, 0)
            if not _across_edge(im_th):
                continue

            cut = Cut2x2(im_th)

            # 0 1
            # 1 0
            if cut.i == 6:
                cuts.extend([
                    Cut2x2([[0, 1], [0, 0]]),
                    Cut2x2([[0, 0], [1, 0]]),
                    Cut2x2([[1, 1], [1, 0]]),
                    Cut2x2([[0, 1], [1, 1]]),
                ])
                check_pattern = cut.i
            # 1 0
            # 0 1
            elif cut.i == 9:
                cuts.extend([
                    Cut2x2([[1, 0], [0, 0]]),
                    Cut2x2([[0, 0], [0, 1]]),
                    Cut2x2([[1, 1], [0, 1]]),
                    Cut2x2([[1, 0], [1, 1]]),
                ])
                check_pattern = cut.i
            else:
                cuts.append(cut)
            # cuts.append(cut)

        cuts = [cut for cut in cuts if cut.ps]

        for j in range(len(cuts) - 1, -1, -1):
            duplicate = cuts[j].i in [cut.i for cut in cuts[:j]]
            if duplicate:
                del cuts[j]

        return cuts, check_pattern

    l_rc = get_rc(0, rr - 1, 0, cc - 1)
    # l_rc = [[18, 10]]

    conts = []

    for r0, c0 in l_rc:
        im = img_b[r0: r0 + 2, c0: c0 + 2]

        # print(im)

        cuts, check_pattern = _pat_bin(im, img_t[r0, c0])

        if check_pattern > 0:
            map_cut_check.append(r0, c0, check_pattern)

        if not cuts:
            continue

        map_cut_2x2.append(r0, c0, cuts)

        for cut in cuts:
            ps = cut.ps
            if len(ps) == 0:
                continue
            conts.append([[c0 + c - 0.5, r0 + r - 0.5] for r, c in ps])

    conts_tree_view.append_plot(Plotter2(conts), tag='edge-2x2')


def cut_is_branch(cut1: Cut2x2, cut2: Cut2x2, de):
    return cut1.di[1 - de] == cut2.di[1 - de]


def cut_is_merge(cut1: Cut2x2, cut2: Cut2x2, de):
    return cut1.di[de] == cut2.di[de]


def cut_is_connected(cut1: Cut2x2, cut2: Cut2x2, de):
    return cut1.di[de] == (cut2.di[1 - de] + 4) % 8


def cut_find_connected(r0, c0, cut0, de, map_cut: ImageMap['list[Cut2x2]']) -> 'tuple[int, int, list[Cut2x2]]':
    i0 = cut0.di[de]
    r1, c1 = get_8d(r0, c0)[i0]

    cuts1 = map_cut.get(r1, c1)
    if cuts1 is None:
        return r1, c1, []

    return r1, c1, [cut for cut in cuts1 if cut_is_connected(cut0, cut, de)]


def find_line_blob(cont):
    cont = np.array(cont)

    lii = []
    for i1 in range(len(cont) - 1):
        li2 = list(range(i1 + 1, len(cont)))
        lk = [distance(cont[i1], cont[i2]) for i2 in li2]

        if len(lk) < 3:
            continue

        lj = []
        for j in range(1, len(lk) - 1):
            if lk[j] <= lk[j - 1] and lk[j] <= lk[j + 1]:
                lj.append(j)

        if len(lj) == 0:
            continue

        j = lj[np.argmin([lk[j] for j in lj])]
        i2 = li2[j]

        lii.append([i1, i2])

    # i2が複数個ある場合に１つに絞る
    dict_ii = dict()
    for i1, i2 in lii:
        if i2 not in dict_ii:
            dict_ii[i2] = [i1]
        else:
            dict_ii[i2].append(i1)

    lii = []
    for i2 in dict_ii:
        li1 = dict_ii[i2]
        lk = [distance(cont[i1], cont[i2]) for i1 in li1]
        i1 = li1[np.argmin(lk)]
        lii.append([i1, i2])
    lii = sorted(lii, key=lambda o: o[0])

    pair_conts = []
    for i1, i2 in lii:
        pair_conts.append([cont[i1], cont[i2]])
    return pair_conts


def plot_edge_peak():
    def _edge(r0, c0):
        if img_p[r0, c0] < 7:
            return

        def _climb_up(r0, c0, r1, c1, d):
            i, _ = quantize_grad_t(img_t[r1, c1])[0]
            i = i + (0 if d > 0 else 4)
            i = i % 8

            ro, co = get_8d()[i]
            r2, c2 = r1 + ro, c1 + co

            if not box_in(r2, c2, rr - 1, cc - 1):
                return 'break', None

            ratio = img_p[r2, c2] / img_p[r0, c0]
            if ratio < 1 / 1.5:
                return 'end', None

            if angle_between_tans(img_t[r0, c0], img_t[r2, c2]) > 1:
                if ratio < 1 / 3:
                    return 'end', None
                # return 'break', None
                return 'end', None

            return 'continue', [r2, c2]

        l_irc = [[0, r0, c0]]

        blank = False
        for d in [1, -1]:

            r1, c1 = r0, c0
            for i in range(2):
                flag, p1 = _climb_up(r0, c0, r1, c1, d)
                if flag == 'continue':
                    r1, c1 = p1
                    l_irc.append([(i + 1) * d, r1, c1])
                    continue
                if flag == 'end':
                    blank = True
                break

        if not blank:
            return False

        # print(l_irc)

        lp = [img_p[r, c] for _, r, c in l_irc]

        if np.max(lp) != lp[0]:
            return False

        # print(lp)

        return True

    def _suppress_edge_pat_diag(r0, c0):
        i, _ = quantize_grad_t(img_t[r0, c0])[0]

        # non-diag
        if i % 2 == 0:
            return False

        ro, co = get_8d()[i]

        le = [img_edge[r0 + ro1, c0 + co1]
              for ro1, co1 in [[ro, co], [ro, 0], [0, co]]]

        # 以下の4つの回転対称パターンのみ
        # 1 1
        # 1 *
        if not (le[1] == 1 and le[2] == 1):
            return False

        lt = [img_t[r0 + ro1, c0 + co1]
              for ro1, co1 in [[0, 0], [ro, 0], [0, co]]]
        t_span = min_interval_of_ps_circular(lt, -np.pi, np.pi)

        return t_span < np.pi / 4

    img_edge = np.zeros((rr, cc))
    l_rc = get_rc(1, rr - 2, 1, cc - 2)
    #l_rc = [[7, 2]]

    map_peak_edge = ImageMap()

    lp = []
    for r, c in l_rc:
        ret = _edge(r, c)
        if not ret:
            continue
        img_edge[r, c] = 1

    for r, c in l_rc:
        if img_edge[r, c] == 0:
            continue

        if _suppress_edge_pat_diag(r, c):
            img_edge[r, c] = 0

    for r, c in np.transpose(np.where(img_edge > 0)):
        mp = np.array([r, c]) + 0.5
        map_peak_edge.append(r, c, mp)
        lp.append(mp)

    lp = np.array(lp)
    plotter = PlotterPoints(lp)
    conts_tree_view.append_plot(plotter, 'edge-peak')

    def _check_contrast(r0, c0, r1, c1):
        # 形状は 2x3
        # 0 0 0
        # 0 0 0

        # if r0 == 12 and c0 == 16 and r1 == 12 and c1 == 17:
        #     print()

        # 斜めはパス
        if abs(r0 - r1) == 1 and abs(c0 - c1) == 1:
            return True

        r0, r1 = sorted([r0, r1])
        c0, c1 = sorted([c0, c1])

        if r1 - r0 == 1:
            im = img_b[r0: r0 + 3, c0: c0 + 2].T
        else:
            im = img_b[r0: r0 + 2, c0: c0 + 3]

        v0 = abs(im[0, 1] - im[1, 1])

        if v0 == 0:
            return False

        v11 = abs(im[0, 0] - im[0, 1])
        v12 = abs(im[0, 1] - im[0, 2])
        v21 = abs(im[1, 0] - im[1, 1])
        v22 = abs(im[1, 1] - im[1, 2])

        v1 = max(v11, v12)
        v2 = max(v21, v22)

        vs = np.array([v1, v2])
        ratios = vs / v0

        return np.any(ratios < 2.5)

    def _proceed(r0, c0, de, only_peak=True):
        t = img_t[r0, c0] + (1 - 2 * de) * np.pi / 2
        li = [i for i, d in quantize_grad_t(t) if d < 0.35]

        l_result = []

        for i in li:
            r1, c1 = get_8d(r0, c0)[i]
            if only_peak and map_peak_edge.get(r1, c1) is None:
                continue

            if not box_in(r1, c1, rr - 1, cc - 1):
                continue

            ratio = calc_ratio(img_p[r0, c0], img_p[r1, c1])
            if ratio < 0.25:
                continue

            t0, t1 = img_t[r0, c0], img_t[r1, c1]
            dt = angle_between_tans(t0, t1)
            score1 = dt / (np.pi / 2)

            ro, co = r1 - r0, c1 - c0
            dx, dy = co, -ro

            # 法線ベクトル
            tt1 = np.arctan2(-dx, dy)
            tt2 = np.arctan2(dx, -dy)

            dt1 = angle_between_tans(tt1, [t0, t1])
            dt2 = angle_between_tans(tt2, [t0, t1])

            # 法線ベクトルからのずれ具合
            dt = max(dt1 if min(list(dt1) + list(dt2)) in dt1 else dt2)
            score2 = dt / (np.pi / 2)

            # 0に近いほど良い
            score = min(score1, score2)

            if score > 0.7:
                continue

            if not _check_contrast(r0, c0, r1, c1):
                continue

            s = ratio - dt
            l_result.append([s, r1, c1])

        return [[r, c] for _, r, c in l_result]

    def _trace_all(cont, N, de):
        c0, r0 = cont[-1]
        while len(cont) < N:
            ret = _proceed(r0, c0, de)
            if len(ret) == 0:
                break

            if len(ret) == 1:
                r0, c0 = ret[0]
                cont.append([c0, r0])
                continue

            conts = []
            for r, c in ret:
                conts.extend(_trace_all(cont + [[c, r]], N, de))
            return conts

        return [cont]

    def _cont_score(cont):
        lp = [img_p[r, c] for c, r in cont]
        lp = np.array(lp) / np.max(lp)
        la = [interior_angle(*cont[i: i + 3]) for i in range(len(cont) - 2)]

        # print(lp, la)

        if np.count_nonzero(lp < 0.3) > 1:
            return 0
        return 1

    def _trace_nonpeak(r0, c0, de, d):
        l = []

        ps1 = _proceed(r0, c0, de, False)
        for r1, c1 in ps1:
            ps2 = _proceed(r1, c1, de)
            for r2, c2 in ps2:
                dr, dc = r2 - r0, c2 - c0
                dx, dy = dc, -dr
                t = np.arctan2(dy, dx)
                l.append([t, [[c1, r1], [c2, r2]]])

        if len(l) == 0:
            return []

        lt = [o[0] for o in l]
        t1, t2 = min_range_of_ps_circular(lt, -np.pi, np.pi)

        # print(r0, c0, t1, t2)

        t = t1 if d == 1 else t2
        cont = [o[1] for o in l if o[0] == t][0]
        return cont

    def _trace(r0, c0, de, d, N=5):
        cont = []
        for _ in range(N):
            ret = _proceed(r0, c0, de)
            if len(ret) == 0:
                cont_nonpeak = _trace_nonpeak(r0, c0, de, d)
                if len(cont_nonpeak) == 0:
                    break
                cont.extend(cont_nonpeak)
                c0, r0 = cont[-1]
                continue

            if len(ret) == 1:
                r0, c0 = ret[0]
                cont.append([c0, r0])
                continue

            #print(r0, c0, ret)

            conts = []
            for r, c in ret:
                conts.extend(_trace_all([[c, r]], 3, de))

            # for _cont in conts:
            #     print(_cont)

            # 分岐点からNピクセル先までを全てトレースしてどの輪郭がベストが評価

            len_max = max([len(cont) for cont in conts])
            conts = [cont for cont in conts if len(cont) >= len_max]

            # スコアで足切り
            ls = np.array([_cont_score([[c0, r0]] + cont) for cont in conts])
            li = np.where(ls >= max(ls) / 2)[0]
            conts = [conts[i] for i in li]

            # 右回り、左回りで選択
            lt = []
            for _cont in conts:
                c1, r1 = _cont[-1]
                dr, dc = r1 - r0, c1 - c0
                dx, dy = dc, -dr
                lt.append(np.arctan2(dy, dx))

            t1, t2 = min_range_of_ps_circular(lt, -np.pi, np.pi)
            t = t1 if d == 1 else t2

            c0, r0 = conts[lt.index(t)][0]
            cont.append([c0, r0])

        return cont

    def _check_1px_short_branch(r0, c0):
        l = []
        for de in [1, 0]:
            ret = _proceed(r0, c0, de)
            base = len(ret) > 0
            rev_branch = False
            for r1, c1 in ret:
                for r2, c2 in _proceed(r1, c1, 1 - de):
                    if r2 == r0 and c2 == c0:
                        continue
                    rev_branch = True
            l.append([base, rev_branch])

        [base_1, rev_branch_1], [base_2, rev_branch_2] = l
        return ((not base_1 and not base_2) or
                (base_1 and rev_branch_1 and not base_2 and not rev_branch_2) or
                (not base_1 and not rev_branch_1 and base_2 and rev_branch_2))

    l_rc = [[r, c] for r, c, _ in map_peak_edge.get_list()]

    l_rem = []
    for r0, c0 in l_rc:
        if _check_1px_short_branch(r0, c0):
            l_rem.append([r0, c0])

    for r, c in l_rem:
        map_peak_edge.remove(r, c)

    lp = [[r + 0.5, c + 0.5] for r, c, _ in map_peak_edge.get_list()]

    plotter = PlotterPoints(lp)
    conts_tree_view.append_plot(plotter, 'edge-peak-2')

    l_rc = [[r, c] for r, c, _ in map_peak_edge.get_list()]
    conts = []

    for r0, c0 in l_rc:
        ret = _proceed(r0, c0, 0) + _proceed(r0, c0, 1)
        for r1, c1 in ret:
            r, c = (r0 + r1) / 2, (c0 + c1) / 2
            conts.append([[c0, r0], [c, r]])

    conts_tree_view.append_cont(conts, tag='edge-peak-conn-8d')

    def _remove_duplicate_point(cont):
        for i in range(len(cont) - 2, -1, -1):
            if cont[i] == cont[i + 1]:
                del cont[i]

    def _dilate_cont(cont, d, i_start, i_end):
        # dj > 2: 鋭角, dj < -2: 鈍角
        def _dj(j1, j2):
            dj = chaincode_dif_clock(j1, j2)
            dj = dj - 8 if dj > 4 else dj
            return -d * dj

        if i_start <= i_end:
            i_start = max([i_start, 0])
            i_end = min([i_end, len(cont)])
            li = range(i_start, i_end - 1)
            i_d = 1
        else:
            i_start = min([i_start, len(cont) - 1])
            i_end = max([i_end, -1])
            li = range(i_start, i_end + 1, -1)
            i_d = -1

        cont_dilate = []

        l_8d = get_8d()
        for i in li:
            (c1, r1), (c2, r2) = cont[i], cont[i + i_d]
            j = l_8d.index([r2 - r1, c2 - c1])

            dj1, dj2 = 0, 0

            im1 = i - i_d
            if im1 >= 0 and im1 < len(cont):
                cm1, rm1 = cont[im1]
                jm1 = l_8d.index([r1 - rm1, c1 - cm1])
                dj1 = _dj(jm1, j)

            i3 = i + 2 * i_d
            if i3 >= 0 and i3 < len(cont):
                c3, r3 = cont[i3]
                j2 = l_8d.index([r3 - r2, c3 - c2])
                dj2 = _dj(j, j2)

            #print(r1, c1, dj1, dj2)

            if dj1 > 2 or dj2 > 2:
                continue

            if j % 2 == 0:
                ro, co = l_8d[(j - d * 2) % 8]
                ps = [[c1 + co, r1 + ro], [c2 + co, r2 + ro]]
                if (i_d > 0 and i == 0) or (i_d < 0 and i == len(cont) - 1) or dj1 >= 2:
                    del ps[0]
                if (i_d > 0 and i == len(cont) - 2) or (i_d < 0 and i == 1) or dj2 >= 2:
                    del ps[-1]

                # ps = [p for p in ps if p not in cont]
                if len(ps) == 0:
                    continue

                # print('a', r1, c1)
                cont_dilate.extend(ps)
            else:
                ro, co = l_8d[(j - d) % 8]
                p = [c1 + co, r1 + ro]
                if p in cont:
                    continue

                # print('s', r1, c1)
                cont_dilate.append(p)

                # 鈍角の場合は回り込むために追加点が必要
                if dj2 <= -2:
                    cont_dilate.append([c2 + co, r2 + ro])

        _remove_duplicate_point(cont_dilate)

        return cont_dilate

    def _find_edge_on_surface(cont_dilate, d):
        l_ret = []

        for i in range(len(cont_dilate)):
            c, r = cont_dilate[i]
            ret = map_peak_edge.get(r, c)
            if ret is None:
                continue

            # deを決める

            l_de = [0, 1]

            i1 = max([i - 1, 0])
            i2 = min([i + 1, len(cont_dilate) - 1])
            p1, p2 = np.array([cont_dilate[i1], cont_dilate[i2]])
            dc, dr = p2 - p1
            dx, dy = dc, -dr

            t_desire = np.arctan2(dy, dx) - d * np.pi / 2

            lt = [img_t[r, c] + (1 - 2 * de) * np.pi / 2 for de in l_de]
            dt = angle_between_tans(t_desire, lt)

            de = l_de[np.argmin(dt)]

            l_ret.append(((r, c), de))

        return l_ret

    def _trace_by_scan2d(r0, c0, d):
        conts_dilate = [[], [], []]
        li_dilate = [0, 0, 0]

        def _pop_back_until(cont, p):
            for i in range(len(cont) - 1, -1, -1):
                if cont[i] == p:
                    break
            return cont[:i + 1]

        def _rollback_dilate_cont_to_junction(cont, i_cont, N_dilate):
            # 分岐を検出した膨張輪郭(N_dilate)以下をロールバック
            # contに近い膨張輪郭から処理

            cont_dilate = cont
            i_dilate = i_cont

            li_dilate[0] = i_cont

            for n_dilate in range(N_dilate + 1):
                ret = _dilate_cont(
                    cont_dilate, d, i_dilate - 1, i_dilate + 1)

                if len(ret) == 0:
                    for n in range(n_dilate, N_dilate + 1):
                        conts_dilate[n].clear()
                        li_dilate[n] = 0
                    break

                conts_dilate[n_dilate] = _pop_back_until(
                    conts_dilate[n_dilate], ret[0])

                cont_dilate = conts_dilate[n_dilate]
                i_dilate = len(cont_dilate) - 1

                n_dilate_1 = n_dilate + 1
                if n_dilate_1 < len(li_dilate):
                    li_dilate[n_dilate_1] = min(
                        [li_dilate[n_dilate_1], i_dilate])

        def _check_branch(cont, cont_branch, i_cont):
            if len(cont_branch) < 2:
                return False

            lp1 = [img_p[r, c] for c, r in cont[i_cont - 2: i_cont + 1]]
            lp2 = [img_p[r, c] for c, r in cont_branch[: 3]]
            # print(cont[i_cont - 2: i_cont + 1], cont_branch[: 3])
            if np.mean(lp1) > np.mean(lp2) * 1.5:
                return False

            d = 0
            for i_cont_branch in range(2):
                i1 = max([i_cont - 1, 0])
                i2 = min([i_cont + 1, len(cont) - 1])
                li = [i1, i_cont, i2]

                p = cont_branch[i_cont_branch]
                ds = [distance(p, cont[i]) for i in li]
                j = np.argmin(ds)

                d = ds[j]
                i_cont = li[j]

            # 分岐点から2px地点でちゃんと分岐しているかチェック
            return d > 1.4

        def _interpolate_point(p1, p2):
            return bresenham(p1, p2)[1: -1]

        def _test_cont(cont):
            l = get_8d()
            for (c1, r1), (c2, r2) in zip(cont[:-1], cont[1:]):
                l.index([r1 - r2, c1 - c2])

        # いまやってるdilate_contと後段のdilate_contが衝突した際に
        # 後段のdilate_contをロールバックする
        # 曲率が高い場合に必要
        def _rollback_n_plus_1_dilate_cont(n_dilate, max_len):
            if n_dilate + 1 >= len(conts_dilate):
                return

            begin = len(conts_dilate[n_dilate]) - 1
            end = max([begin - max_len, -1])
            l_ii = itertools.product(range(begin, end, -1),
                                     range(len(conts_dilate[n_dilate + 1])))

            for i1, i2 in l_ii:
                p1 = conts_dilate[n_dilate][i1]
                p2 = conts_dilate[n_dilate + 1][i2]
                if p1 != p2:
                    continue

                del conts_dilate[n_dilate + 1][i2:]
                li_dilate[n_dilate + 1] = i1

                #print(n_dilate, 'akdjal', i1, i2, p1)
                return

        def _dilate_cont_and_make_branch(cont):
            branched = False

            de = 0  # dummy

            cont_dilate = cont

            for n_dilate in range(3):
                # print(n_dilate)

                i_dilate = li_dilate[n_dilate]

                # ぎりぎりまでやれば len(cont_dilate) だが、
                # 曲率が高い時にはみ出してしまわないように
                i_end = len(cont_dilate) - 3
                li_dilate[n_dilate] = max(i_end - 1, 0)

                try:
                    cont_dilate = _dilate_cont(
                        cont_dilate, d, i_dilate, i_end)
                except Exception as e:
                    print('_dilate_cont error', e, cont_dilate, d, i_dilate)
                    break

                max_len = len(cont_dilate)

                conts_dilate[n_dilate].extend(cont_dilate)
                _remove_duplicate_point(conts_dilate[n_dilate])

                _rollback_n_plus_1_dilate_cont(n_dilate, max_len)

                try:
                    _test_cont(conts_dilate[n_dilate])
                except Exception as e:
                    print('dilate error', e, n_dilate, conts_dilate[n_dilate])
                    break
                #print(n_dilate, li_dilate[n_dilate], conts_dilate[n_dilate])

                l_ret = _find_edge_on_surface(cont_dilate, d)
                cont_dilate = conts_dilate[n_dilate]

                for ret in l_ret:
                    [r0, c0], de = ret
                    cont1 = _trace(r0, c0, de, d)

                    #print('a', r0, c0, n_dilate, de, cont1)

                    # contで最近傍点を探す

                    # 閉じていた場合に先頭付近とマッチングしてしまうのを防ぐ
                    i_start = 2
                    lk = [np.linalg.norm([r - r0, c - c0])
                          for c, r in cont[i_start:]]
                    i = np.argmin(lk) + i_start

                    if not _check_branch(cont, [[c0, r0]] + cont1, i):
                        continue

                    # print('s', r0, c0, n_dilate, de, cont1)

                    # 分岐発生
                    branched = True

                    _rollback_dilate_cont_to_junction(cont, i, n_dilate)

                    ps_interp = _interpolate_point(cont[i], [c0, r0])
                    cont = cont[:i + 1] + ps_interp + [[c0, r0]] + cont1
                    break

                if branched:
                    break

            return branched, (de, cont)

        def _close_cont(cont):
            p0 = cont[0]
            ld = [0]
            for i in range(1, len(cont)):
                p1 = cont[i]
                d = distance(p0, p1)
                ld.append(d)
                if d == 0:
                    return True, cont[:i]

            for i in range(2, len(cont) - 1):
                if ld[i] > 1.5:
                    continue
                if ld[i] <= ld[i - 1] and ld[i] <= ld[i + 1]:
                    return True, cont[:i + 1]

            return False, cont

        def _detect_line_blob(cont):
            cont = np.array(cont)

            i1 = len(cont) - 1
            li2 = list(range(max(i1 - 10, 0), i1))
            lk = [distance(cont[i1], cont[i2]) for i2 in li2]

            if len(lk) < 3:
                return False

            lj = []
            for j in range(1, len(lk) - 1):
                if lk[j] > 3:
                    continue
                if lk[j] <= lk[j - 1] and lk[j] <= lk[j + 1]:
                    lj.append(j)

            if len(lj) == 0:
                return False

            j = lj[np.argmin([lk[j] for j in lj])]
            i2 = li2[j]

            print(cont[i1], cont[i2])

            return True

        de = 1

        cont = [[c0, r0]]
        debug_conts = []

        step = 0
        max_step = 3

        while step < max_step:
            step += 1

            cont1 = _trace(r0, c0, de, d)
            if len(cont1) == 0:
                print('trace end')
                break

            cont = cont + cont1

            if _detect_line_blob(cont):
                print('detect_line_blob')
                break

            closed, cont = _close_cont(cont)
            if closed:
                print('closed')
                break

            while step < max_step:
                step += 1
                branched, info = _dilate_cont_and_make_branch(cont)
                if not branched:
                    break
                de, cont = info

            closed, cont = _close_cont(cont)
            if closed:
                print('closed')
                break

            c0, r0 = cont[-1]

        debug_conts.extend(conts_dilate)

        return cont, debug_conts

    class Self:
        def __init__(self):
            self.conts_dilate = [[] for _ in range(3)]
            self.li_dilate = [[0, 0] for _ in range(3)]
            self.img_done_dilate = np.zeros((rr, cc))

    self = Self()

    # d_dilate: 0 or 1
    def _dilate_cont_and_find_pair(cont, d_dilate, d):
        found_pair = False
        l_ret = []  # return

        cont_dilate = cont

        conts_dilate = self.conts_dilate
        li_dilate = self.li_dilate
        img_done_dilate = self.img_done_dilate

        for n_dilate in range(3):
            # print(n_dilate)

            i_d = -1 if d_dilate == 1 else 1

            # 初回
            if li_dilate[n_dilate][0] == li_dilate[n_dilate][1]:
                if d_dilate == 1:
                    i_start = 0
                    i_end = len(cont_dilate)
                else:
                    i_start = len(cont_dilate) - 1
                    i_end = -1
                li_dilate[n_dilate][0] = 0
                li_dilate[n_dilate][1] = len(cont_dilate) - 1

            # 2回目
            else:
                i_start = max(li_dilate[n_dilate][d_dilate] + i_d, 0)
                i_end = len(cont_dilate) if d_dilate == 1 else -1
                li_dilate[n_dilate][d_dilate] = i_end + i_d

            try:
                cont_dilate = _dilate_cont(
                    cont_dilate, d, i_start, i_end)
            except Exception as e:
                print('_dilate_cont error_1', e, cont_dilate,
                      n_dilate, d, i_start, i_end)
                break

            prev_len = len(conts_dilate[n_dilate])

            if d_dilate == 1:
                conts_dilate[n_dilate].extend(cont_dilate)
            else:
                conts_dilate[n_dilate] = cont_dilate[::-1] + \
                    conts_dilate[n_dilate]

            _remove_duplicate_point(conts_dilate[n_dilate])

            try:
                _test_cont(conts_dilate[n_dilate])
            except Exception as e:
                print('_dilate_cont error_2', e, cont_dilate,
                      n_dilate, d, i_start, i_end)
                break

            if d_dilate == 0 and n_dilate + 1 < len(li_dilate):
                i_offset = len(conts_dilate[n_dilate]) - prev_len
                for _ in [0, 1]:
                    li_dilate[n_dilate + 1][_] += i_offset

            l_ret = []
            for c0, r0 in cont_dilate:
                if not box_in(r0, c0, rr - 1, cc - 1):
                    continue

                if img_done_dilate[r0, c0] == 1:
                    continue
                img_done_dilate[r0, c0] = 1

                if map_peak_edge.get(r0, c0) is None:
                    continue

                ret1, ret2 = _proceed(r0, c0, 0), _proceed(r0, c0, 1)
                if len(ret1) == 0 or len(ret2) == 0:
                    continue

                if [c0, r0] in cont:
                    continue

                found_pair = True
                l_ret.append([r0, c0])

            cont_dilate = conts_dilate[n_dilate]

            if found_pair:
                break

        return found_pair, l_ret, n_dilate

    def _test_cont(cont):
        l = get_8d()
        for (c1, r1), (c2, r2) in zip(cont[:-1], cont[1:]):
            l.index([r1 - r2, c1 - c2])

    def _update_li_dilate(size):
        for _ in [0, 1]:
            self.li_dilate[0][_] += size

    def _update_img_done_dilate(cont):
        for c, r in cont:
            self.img_done_dilate[r, c] = 1

    class Method(Enum):
        TRACE_BACK = 1
        TRACE_FRONT = 2
        DILATE_BACK = 3
        DILATE_FRONT = 4
        TRACE_PAIR_MAIN_BACK = 5
        TRACE_PAIR_SUB_BACK = 6
        TRACE_PAIR_MAIN_FRONT = 7
        TRACE_PAIR_SUB_FRONT = 8

    def _main_trace_pair(r0, c0, de, d, debug_print=True):
        def _trace_back_internal():
            c0, r0 = conts[-1][-1]
            de, d = l_ds['back']
            new_cont = _trace(r0, c0, de, d)
            conts[-1] = conts[-1] + new_cont
            # duplicate_count = _count_duplicate(new_cont, d)
            # and duplicate_count <= len(new_cont) - 2
            return len(new_cont) > 0

        def _trace_front_internal():
            c0, r0 = conts[-1][0]
            de, d = l_ds['front']
            new_cont = _trace(r0, c0, de, d)
            conts[-1] = new_cont[::-1] + conts[-1]
            # duplicate_count = _count_duplicate(new_cont, d)
            if len(new_cont) > 0:  # and duplicate_count <= len(new_cont) - 2:
                _update_li_dilate(len(new_cont))
                return True
            return False

        def _count_duplicate(cont, d):
            done_d = done_d = 0b10 if d == 1 else 0b01
            count = 0
            for c, r in cont:
                if img_done[r, c] & done_d:
                    continue
                count += 1
            return count

        def _trace_back():
            if _trace_back_internal():
                run_queue.extend([Method.DILATE_BACK, Method.TRACE_BACK])

        def _trace_front():
            if _trace_front_internal():
                run_queue.extend([Method.DILATE_FRONT, Method.TRACE_FRONT])

        def _dilate_back():
            _, d = l_ds['back']
            found_pair, l_rc, n_dilate = _dilate_cont_and_find_pair(
                conts[-1], d_dilate=1, d=d)
            if not found_pair:
                return

            if debug_print:
                print('found!!', *l_rc)

            ret = None
            for r1, c1 in l_rc:
                ret = _decide_whether_pair_or_branch(
                    r1, c1, Method.DILATE_BACK, n_dilate)
                if ret is None:
                    continue
                break

            if ret is None:
                return

            if ret == 'pair':
                conts_pair.append([[c1, r1]])
                run_queue[0: 0] = [Method.TRACE_PAIR_SUB_BACK,
                                   Method.TRACE_PAIR_SUB_FRONT]

            _insert_dilate(Method.TRACE_BACK, Method.DILATE_BACK)

        def _dilate_front():
            _, d = l_ds['front']
            found_pair, l_rc, n_dilate = _dilate_cont_and_find_pair(
                conts[-1], d_dilate=0, d=d)
            if not found_pair:
                return

            if debug_print:
                print('found!!', *l_rc)

            ret = None
            for r1, c1 in l_rc:
                ret = _decide_whether_pair_or_branch(
                    r1, c1, Method.DILATE_FRONT, n_dilate)
                if ret is None:
                    continue
                break

            if ret is None:
                return

            if ret == 'pair':
                conts_pair.append([[c1, r1]])
                run_queue[0: 0] = [Method.TRACE_PAIR_SUB_BACK,
                                   Method.TRACE_PAIR_SUB_FRONT]

            _insert_dilate(Method.TRACE_FRONT, Method.DILATE_FRONT)

        def _decide_whether_pair_or_branch(r0, c0, dilate_type, n_dilate):
            cont = conts[-1]

            lk = np.linalg.norm(np.array(cont) - [c0, r0], axis=1)
            li = np.where(lk == lk.min())[0]
            i = li[0] if dilate_type == Method.DILATE_BACK else li[-1]

            c1, r1 = cont[i]
            dt = angle_between_tans(img_t[r0, c0], img_t[r1, c1])
            rev = dt > np.pi * 0.6

            if dilate_type == Method.DILATE_BACK:
                ds_key = 'back'
            elif dilate_type == Method.DILATE_FRONT:
                ds_key = 'front'
            else:
                raise Exception()

            de, d = l_ds[ds_key]

            if rev:
                ds = 1 - de, -d
            else:
                ds = de, -d

            de1, d1 = ds
            de2, d2 = 1 - de1, -d1
            ret1 = _trace(r0, c0, de1, d1)
            ret2 = _trace(r0, c0, de2, d2)
            ret = ret1[::-1] + [[c0, r0]] + ret2

            # 高曲率の場合遠く離れた場所とマッチしてしまう
            sub_cont = cont[max(i - 7, 0): min(i + 8, len(cont))]
            l_sli, _, l_d = suture_line_i(sub_cont, ret, th_k=3)

            # print(sub_cont)
            # print(ret)
            # print(l_sli)
            # print(l_d)
            # print(len(ret1))

            li = l_sli[:, 1][np.where(l_d > 0)]
            if len(li) > 0 and max(li) - min(li) >= 3:
                return 'pair'

            # 釣り針パターン対策
            # -------
            #    |   | <-こっち側とマッチしている
            #    ----
            j = len(ret1)
            if np.count_nonzero(l_sli[:, 1] == j) == 0:
                return

            li = l_sli[:, 1]
            if 0 not in li and len(ret) - 1 in li:
                new_cont = ret1
                new_ds = [de1, d]
            elif 0 in li and len(ret) - 1 not in li:
                new_cont = ret2
                new_ds = [de2, d]
            else:
                return

            # 両方向に伸びている場合は分岐ではない
            # li = l_sli[:,1]
            # if (len(ret1) >= 2 and len(np.where(li < len(ret1))[0]) <= 2 and
            #     len(ret2) >= 2 and len(np.where(li > len(ret1))[0]) <= 2):
            #     return

            # if j == 0 or np.any(l_d[:j] == 0):
            #     new_cont = ret2
            #     new_ds = [de2, d]
            # else:
            #     new_cont = ret1
            #     new_ds = [de1, d]

            if not _check_branch(cont, new_cont, i):
                return

            l_ds[ds_key] = new_ds

            d_dilate = 1 if dilate_type == Method.DILATE_BACK else 0
            _rollback_dilate_cont_to_junction(cont, i, n_dilate, d_dilate)

            ps_interp = bresenham(cont[i], [c0, r0])[1: -1]
            if dilate_type == Method.DILATE_BACK:
                conts[-1] = cont[:i + 1] + ps_interp + [[c0, r0]] + new_cont
            else:
                prev_len = len(conts[-1])
                conts[-1] = new_cont[::-1] + [[c0, r0]] + ps_interp + cont[i:]
                _update_li_dilate(len(conts[-1]) - prev_len)

            return 'branch'

        def _check_branch(cont, cont_branch, i_cont):
            if len(cont_branch) <= 2:
                return False

            lp1 = [img_p[r, c] for c, r in cont[i_cont - 2: i_cont + 1]]
            lp2 = [img_p[r, c] for c, r in cont_branch[: 3]]
            # print(cont[i_cont - 2: i_cont + 1], cont_branch[: 3])
            if np.mean(lp2) > 10:
                return True
            if np.mean(lp1) > np.mean(lp2) * 1.5:
                return False

        def _rollback_dilate_cont_to_junction(cont, i_cont, N_dilate, d_dilate):
            # 分岐を検出した膨張輪郭(N_dilate)以下をロールバック
            # contに近い膨張輪郭から処理

            cont_dilate = cont
            i_dilate = i_cont

            li_dilate = self.li_dilate
            conts_dilate = self.conts_dilate

            li_dilate[0][d_dilate] = i_cont

            for n_dilate in range(N_dilate + 1):
                if d_dilate == 1:
                    i_start = i_dilate - 1
                    i_end = i_dilate + 1
                else:
                    i_start = i_dilate
                    i_end = i_dilate + 2

                _, d = l_ds['back']
                ret = _dilate_cont(
                    cont_dilate, d, i_start, i_end)

                if len(ret) == 0:
                    # よくわからないけど、とりあえずClear
                    for n in range(n_dilate, N_dilate + 1):
                        conts_dilate[n].clear()
                        li_dilate[n] = [0, 0]
                    break

                prev_len = len(conts_dilate[n_dilate])

                _pop = _pop_back_until if d_dilate == 1 else _pop_front_until
                conts_dilate[n_dilate] = _pop(conts_dilate[n_dilate], ret[0])

                cont_dilate = conts_dilate[n_dilate]
                i_dilate = len(cont_dilate) - 1 if d_dilate == 1 else 0

                n_dilate_1 = n_dilate + 1
                if n_dilate_1 < len(li_dilate):
                    min_or_max = min if d_dilate == 1 else max
                    li_dilate[n_dilate_1][d_dilate] = min_or_max(
                        [li_dilate[n_dilate_1][d_dilate], i_dilate])

                if d_dilate == 0 and n_dilate + 1 < len(li_dilate):
                    i_offset = len(conts_dilate[n_dilate]) - prev_len
                    for _ in [0, 1]:
                        li_dilate[n_dilate + 1][_] += i_offset

        def _pop_back_until(cont, p):
            for i in range(len(cont) - 1, -1, -1):
                if cont[i] == p:
                    break
            return cont[:i + 1]

        def _pop_front_until(cont, p):
            for i in range(len(cont)):
                if cont[i] == p:
                    break
            return cont[i:]

        def _insert_dilate(trace, dilate):
            try:
                i = run_queue.index(trace)
            except:
                i = len(run_queue)
            run_queue.insert(i, dilate)

        def _check_closed(l_sli):
            # ２列目が逆流した場合

            # backの場合
            # [[0, 0]
            #  [1, 1]
            #  [2, 2]
            #  [3, 4]  ↓ ここから
            #  [4, 3]]

            # frontの場合
            # [[0, 1]
            #  [1, 0] ↑ ここから
            #  [2, 2]
            #  [3, 3]
            #  [4, 4]]

            # 処理は共通

            l = l_sli[:, 1]
            l = l[1:] - l[:-1]
            l = np.where(l < 0)[0]
            return len(l) > 0

        def _suture_back_internal(finish: bool):
            def _finish(i_stop):
                _update_img_done_dilate(conts_pair[-1])
                conts_pair[-1] = conts_pair[-1][:i_stop + 1]

            l_sli, l_slp, l_d = suture_line_i(
                conts[-1], conts_pair[-1], th_k=3.1)

            i1, i2 = l_sli[-1]
            di1 = len(conts[-1]) - 1 - i1
            di2 = len(conts_pair[-1]) - 1 - i2

            if finish or (di1 > 3 and di2 > 3):
                _finish(i2)
                return

            if _check_closed(l_sli):
                _finish(l_sli[-1, 1])
                _try_remove_trace(Method.TRACE_BACK)  # Backは閉じた
                return

            if di1 < di2:
                run_queue.insert(0, Method.TRACE_PAIR_MAIN_BACK)
            else:
                run_queue.insert(0, Method.TRACE_PAIR_SUB_BACK)

        def _suture_front_internal(finish: bool):
            def _finish(i_stop):
                _update_img_done_dilate(conts_pair[-1])
                conts_pair[-1] = conts_pair[-1][i_stop:]
            l_sli, l_slp, l_d = suture_line_i(
                conts[-1], conts_pair[-1], th_k=3.1)
            # print(l_slp)

            i1, i2 = l_sli[0]
            di1, di2 = i1, i2

            if finish or (di1 > 3 and di2 > 3):
                _finish(i2)
                return

            # check closed

            if _check_closed(l_sli):
                _finish(l_sli[0, 1])
                _try_remove_trace(Method.TRACE_FRONT)  # Frontは閉じた
                return

            if di1 < di2:
                run_queue.insert(0, Method.TRACE_PAIR_MAIN_FRONT)
            else:
                run_queue.insert(0, Method.TRACE_PAIR_SUB_FRONT)

        def _try_remove_trace(trace):
            try:
                run_queue.remove(trace)
            except:
                pass

        def _trace_pair_main_back():
            has_next = _trace_back_internal()
            _suture_back_internal(not has_next)

        def _trace_pair_sub_back():
            c0, r0 = conts_pair[-1][-1]
            de, d = l_ds['back']
            new_cont = _trace(r0, c0, 1 - de, -d)
            conts_pair[-1] = conts_pair[-1] + new_cont
            has_next = len(new_cont) > 0
            _suture_back_internal(not has_next)

        def _trace_pair_main_front():
            has_next = _trace_front_internal()
            _suture_front_internal(not has_next)

        def _trace_pair_sub_front():
            c0, r0 = conts_pair[-1][0]
            de, d = l_ds['front']
            new_cont = _trace(r0, c0, de, d)
            conts_pair[-1] = new_cont[::-1] + conts_pair[-1]
            has_next = len(new_cont) > 0
            _suture_front_internal(not has_next)

        def _close_cont(cont):
            p0 = cont[0]
            ld = [0]
            for i in range(1, len(cont)):
                p1 = cont[i]
                d = distance(p0, p1)
                ld.append(d)
                if d == 0:
                    return True, cont[:i]

            for i in range(2, len(cont) - 1):
                if ld[i] > 1.5:
                    continue
                if ld[i] <= ld[i - 1] and ld[i] <= ld[i + 1]:
                    return True, cont[:i + 1]

            return False, cont

        method_dict = {
            Method.TRACE_BACK: _trace_back,
            Method.TRACE_FRONT: _trace_front,
            Method.DILATE_BACK: _dilate_back,
            Method.DILATE_FRONT: _dilate_front,
            Method.TRACE_PAIR_MAIN_BACK: _trace_pair_main_back,
            Method.TRACE_PAIR_SUB_BACK: _trace_pair_sub_back,
            Method.TRACE_PAIR_MAIN_FRONT: _trace_pair_main_front,
            Method.TRACE_PAIR_SUB_FRONT: _trace_pair_sub_front,
        }

        conts = [[[c0, r0]]]  # 関数内から変更するためにcontをcontsにしている
        conts_pair = []

        l_ds = {'front': [1 - de, -d], 'back': [de, d]}

        run_queue = [Method.TRACE_BACK, Method.TRACE_FRONT]

        max_step = 40

        debug = False
        if debug:
            max_step = 1

        for _ in range(max_step):
            if len(run_queue) == 0:
                if debug_print:
                    print('no runnable.')
                break

            method = run_queue.pop(0)
            if debug_print:
                print('main_loop = {:2}, method = {}'.format(_, method))

            method_dict.get(method)()

            closed, cont = _close_cont(conts[-1])
            if closed:
                conts[-1] = cont
                if debug_print:
                    print('closed.')
                break

        if False:
            print('')
            print('self.conts_dilate =', self.conts_dilate)
            print('self.li_dilate =', self.li_dilate)
            print('self.img_done_dilate[{}] = 1'.format(
                tuple([o.tolist() for o in np.where(self.img_done_dilate)])))
            print('conts =', conts)
            print('conts_pair =', conts_pair)
            print('l_ds =', l_ds)
            print('run_queue = [{}]'.format(
                ', '.join([str(o) for o in run_queue])))

        conts_pair = [cont for cont in conts_pair if len(cont) > 1]

        return conts[-1], conts_pair

    l_rcd = [[5, 14, 1], [14, 10, -1], [6, 3, -1]]
    l_rcd = [[19, 8, -1]]

    # r0, c0 = 9, 11 # 118

    l_rcd = []
    for r, c, _ in map_peak_edge.get_list():
        l_rcd.extend([[r, c, 1], [r, c, -1]])

    # 分岐でデバッグ用
    # l_rcd = [[6, 23, 1]] # max_step = 3
    # l_rcd = [[4, 16, 1]] # max_step = 13
    # trace_pair at: r = 18, c = 17, d = -1, count = 40

    # あれ？分岐しないの？
    # trace_pair at: r = 3, c = 17, d = 1, count = 11

    # ペアでダブり
    # l_rcd = [[2, 23, -1]]
    # l_rcd = [[4, 1, -1]]

    # １ピクセルで[2, 23, -1]とダブり
    # l_rcd = [[3, 15, -1]]

    # ペア関連で謎の挙動
    # trace_pair at: r = 4, c = 23, d = -1, count = 13
    # trace_pair at: r = 6, c = 1, d = 1, count = 14
    # trace_pair at: r = 6, c = 23, d = -1, count = 17

    # l_rcd = [[30, 19, 1]]

    # l_rcd = []

    img_done = np.zeros((rr, cc), dtype=np.uint8)
    count = 0
    l_conts = []
    main_conts = []

    for r0, c0, d in l_rcd:
        done_d = 0b10 if d == 1 else 0b01

        if img_done[r0, c0] & done_d:
            continue

        # if count >= 5:
        #     break
        # count += 1

        print('l_rcd = [[{}, {}, {}]], count = {}'.format(r0, c0, d, count))

        self = Self()
        cont, conts_pair = _main_trace_pair(r0, c0, 1, d, debug_print=False)

        new_conts = []

        for i, cont in enumerate([cont] + conts_pair):
            li = np.zeros(len(cont), np.bool)
            for i, (c, r) in enumerate(cont):
                if img_done[r, c] & done_d:
                    continue
                img_done[r, c] += done_d
                li[i] = True
            l_li = connect_line_bb(li, tolerance=2)
            l_li = [li for li in l_li if len(li) > 1]

            conts = [cont[li[0]: li[-1] + 1] for li in l_li]
            new_conts.extend([cont[li[0]: li[-1] + 1] for li in l_li])
            main_conts.extend(conts)
            # new_conts.append(cont)

        if len(new_conts) > 0:
            l_conts.append(new_conts)
            count += 1

    #plotter = Plotter(conts, r=0.5, c=0.5, marker='o-')
    conts_tree_view.append_cont(l_conts, tag='edge-peak-conn')
    conts_tree_view.append_cont(self.conts_dilate, tag='edge-peak-conn debug')

    # print('main_conts =', main_conts)
    # main_conts = [[[29, 12], [29, 11], [28, 10], [27, 9], [27, 8], [26, 7], [26, 6], [26, 5], [25, 4], [24, 3], [23, 2], [22, 2], [21, 2], [20, 2], [19, 1], [18, 1], [17, 1], [16, 2], [15, 2], [14, 3], [13, 3], [12, 3], [11, 3], [10, 4], [9, 4], [8, 4], [7, 4], [6, 5], [5, 5], [4, 5], [3, 4], [2, 4], [1, 4]], [[10, 2], [9, 2], [8, 2], [7, 3], [6, 2], [5, 2], [4, 3], [3, 3]], [[22, 2], [21, 2], [20, 2], [19, 1], [18, 1], [17, 1], [16, 2], [15, 2], [14, 3], [13, 3], [12, 3], [11, 3], [10, 4], [9, 4], [8, 4], [7, 4], [6, 5], [5, 5], [4, 6], [5, 7], [6, 7]], [[21, 3], [20, 3], [19, 3], [18, 3], [17, 3], [16, 4], [15, 4], [14, 4], [13, 4], [12, 5], [11, 5], [10, 5], [9, 6], [8, 6], [7, 6]], [[25, 1], [25, 2]], [[25, 1], [25, 2]], [[27, 2], [27, 1], [28, 1], [29, 1]], [[27, 2], [27, 1], [28, 1], [29, 1]], [[3, 3], [4, 3], [5, 2], [6, 2], [7, 3], [8, 2], [9, 2], [10, 2], [11, 2]], [[29, 12], [29, 11], [28, 10], [27, 9], [27, 8], [26, 7], [26, 6], [26, 5], [25, 4], [24, 3], [23, 2]], [[1, 9], [2, 9], [3, 9], [4, 8], [5, 7], [6, 7], [7, 6], [8, 6], [9, 6], [10, 5], [11, 5], [12, 5], [13, 4], [14, 4], [15, 4], [16, 4], [17, 3], [18, 3], [19, 3], [20, 3], [21, 3], [22, 4], [23, 4], [23, 5], [22, 6], [21, 6], [20, 6], [19, 7], [18, 7], [17, 7], [16, 8], [15, 9], [14, 9], [13, 10], [12, 10], [11, 10], [10, 10], [9, 11]], [[5, 10], [6, 10], [7, 10], [8, 10], [9, 9], [10, 9], [11, 9]], [[4, 5], [3, 4], [2, 4], [1, 4]], [[23, 4], [23, 5], [22, 6], [21, 6], [20, 6], [19, 7], [18, 7], [17, 7], [16, 8], [15, 9], [14, 9], [13, 10], [12, 10], [11, 10], [10, 10], [9, 11], [8, 12]], [[19, 9], [18, 10], [17, 10], [16, 10], [15, 11], [14, 11], [13, 12], [12, 12], [11, 12], [10, 13], [9, 13]], [[1, 6], [2, 6], [3, 6], [4, 7], [4, 8], [4, 9]], [[4, 6], [3, 7]], [[5, 12], [6, 12], [7, 12], [8, 11]], [[1, 6], [2, 6], [3, 6], [4, 7]], [[8, 11], [7, 12], [6, 12], [5, 12]], [[3, 12], [2, 12], [1, 12]], [[3, 7], [2, 7]], [[4, 8], [3, 9], [2, 9]], [[14, 11], [15, 11], [16, 10], [17, 10], [18, 10], [19, 9], [20, 9], [21, 9], [22, 8], [23, 8], [23, 7]], [[7, 13], [8, 13]], [[4, 9], [5, 10], [6, 10], [7, 10], [8, 10], [9, 9], [10, 9], [11, 9], [12, 8]], [[21, 9], [22, 8], [23, 8]], [[8, 12], [9, 13], [10, 13], [11, 12], [12, 12], [13, 12]], [[21, 10], [22, 11], [23, 11], [24, 12], [25, 13], [26, 13], [27, 14], [28, 15], [29, 16]], [[22, 10], [22, 11], [23, 11], [24, 12], [25, 13], [26, 13], [27, 14]], [[4, 19], [4, 18], [4, 17], [4, 16], [4, 15], [4, 14], [5, 13], [4, 12], [3, 12], [2, 12], [1, 12]], [[2, 15], [2, 14]], [[26, 17], [25, 17], [24, 16], [23, 16], [22, 16], [21, 15], [20, 14], [19, 14], [18, 14], [17, 14], [16, 15], [15, 15], [14, 16], [13, 16], [12, 17], [11, 17], [10, 17], [9, 17], [8, 17], [7, 18], [6, 18], [5, 18], [4, 17], [4, 16], [4, 15], [4, 14], [5, 13]], [[7, 13], [8, 13]], [[2, 14], [2, 15], [2, 16], [2, 17], [2, 18], [3, 19], [4, 19], [4, 18]], [[26, 18], [25, 17], [24, 16], [23, 16], [22, 16], [21, 15], [20, 14], [19, 14], [18, 14], [17, 14], [16, 15], [15, 15], [14, 16], [13, 16], [12, 17], [11, 17], [10, 17], [9, 17], [8, 17], [7, 18], [6, 18], [5, 18]], [[14, 18], [15, 18], [14, 18], [13, 18], [12, 19], [11, 19], [10, 19], [9, 19], [8, 19]], [[2, 18], [2, 19], [2, 18], [2, 17], [2, 16]], [[28, 15], [29, 16]], [[8, 16], [9, 16]], [[8, 16], [9, 16]], [[1, 18], [2, 18], [3, 19]], [[8, 19], [9, 19], [10, 19], [11, 19], [12, 19], [13, 18], [14, 18], [15, 18], [16, 18], [17, 18], [18, 18], [19, 19], [20, 19], [21, 19], [22, 19], [23, 20], [24, 20], [25, 20], [26, 21], [27, 22], [28, 23], [29, 23]], [[14, 21], [15, 21], [16, 21], [17, 21], [18, 21], [19, 21], [20, 22], [21, 22]], [[17, 18], [18, 18], [19, 19], [20, 19], [21, 19], [22, 19], [22, 18], [22, 17]], [[2, 19], [1, 20]], [[1, 27], [1, 26], [1, 25], [2, 24], [3, 24], [4, 25], [5, 25], [6, 24], [5, 23], [5, 22], [5, 21], [4, 20]], [[5, 26], [6, 25], [7, 26]], [[2, 29], [2, 28], [1, 27], [1, 26], [1, 25], [1, 24], [2, 24], [3, 24], [4, 25], [5, 25], [6, 24], [5, 23], [5, 22], [5, 21], [4, 20]], [[19, 21], [18, 21], [17, 21], [16, 21], [15, 21], [14, 21], [13, 22], [13, 23], [12, 24], [11, 24], [10, 25]], [[16, 23], [15, 23], [14, 24], [13, 25], [12, 26], [11, 26], [10, 26]], [[1, 21], [2, 22], [2, 23]], [[7, 27], [8, 28], [8, 29]], [[1, 21], [2, 22], [2, 23]], [[9, 23], [10, 23], [11, 22], [12, 22]], [[9, 23], [10, 23], [11, 22], [12, 22]], [[23, 25], [23, 24], [22, 23]], [[13, 22], [13, 23], [12, 24], [11, 24], [10, 25]], [[23, 25], [23, 24], [22, 23], [21, 22], [20, 22], [19, 22]], [[23, 23], [22, 22]], [[23, 23], [22, 22]], [[2, 29], [2, 28]], [[1, 24], [1, 23]], [[11, 29], [10, 29], [9, 28], [9, 27], [10, 26], [11, 26], [12, 26], [13, 25], [14, 24], [15, 23], [16, 23], [17, 24], [18, 25], [18, 26], [19, 26], [20, 27], [21, 27], [22, 27], [23, 27], [24, 27], [25, 28], [26, 29], [27, 29]], [[27, 23], [28, 23], [29, 23]], [[17, 24], [18, 25], [19, 25], [19, 26], [20, 27], [21, 27], [22, 27], [23, 27], [24, 27], [25, 28], [26, 29], [27, 29]], [[6, 29], [6, 28], [5, 27], [5, 26]], [[6, 25], [7, 26], [7, 27], [8, 28], [8, 29]], [[14, 26], [15, 26], [16, 26], [17, 26]], [[14, 29], [15, 29], [16, 29], [17, 29]], [[11, 29], [10, 29], [9, 28], [9, 27]], [[14, 26], [15, 26], [16, 26], [17, 26], [18, 26]], [[6, 29], [6, 28], [5, 27]], [[17, 29], [16, 29], [15, 29], [14, 29], [13, 29]], [[24, 28], [25, 29]], [[24, 28], [25, 29]]]

    conts_tree_view.append_cont(main_conts, tag='edge-peak-main-conn')

    l_pair_conts = []
    for i1, i2 in indices_nC2(len(main_conts)):
        cont1 = main_conts[i1]
        cont2 = main_conts[i2]
        l_sli, _, l_d = suture_line_i(cont1, cont2, th_k=5)

        if len(l_sli) < 3:
            continue

        li1, li2 = np.transpose(l_sli)
        mn1, mx1 = min(li1), max(li1)
        mn2, mx2 = min(li2), max(li2)

        if mx1 - mn1 < 6 or mx2 - mn2 < 6:
            continue

        d_mean = np.mean(l_d)
        if d_mean < 1:
            continue
        if len(l_sli) < d_mean:
            continue

        l_pair_conts.append([
            cont1[mn1: mx1 + 1],
            cont2[mn2: mx2 + 1],
        ])

    conts_tree_view.append_cont(l_pair_conts, tag='edge-peak-pair-conn')


def plot_peak_line():
    def _is_peak(r0, c0):
        i = np.argmax([abs(img_b[r, c] - img_b[r0, c0])
                       for r, c in get_8d(r0, c0)])
        r1, c1 = get_8d(r0, c0)[i]
        r2, c2 = get_8d(r0, c0)[(i + 4) % 8]

        lv = np.array([img_b[r1, c1], img_b[r0, c0], img_b[r2, c2]])
        ld = lv[1:] - lv[:-1]
        if ld[0] * ld[1] >= 0:
            return False

        ld = np.abs(ld)
        th = 0.5 if np.max(ld) < 10 else 0.6
        if calc_ratio(*ld) < th:
            return False

        # 鞍点回避
        im = img_b[r0 - 1: r0 + 2, c0 - 1: c0 + 2] - img_b[r0, c0]
        im = im / im[r1 - r0 + 1, c1 - c0 + 1]
        return abs(np.min(im)) < 0.4

    points = []

    l_rc = get_rc(1, rr - 1, 1, cc - 1)
    # l_rc = [[11, 27]]

    for r0, c0 in l_rc:
        if _is_peak(r0, c0):
            points.append([r0, c0])

    plotter = PlotterPoints(points, cl='b')
    conts_tree_view.append_plot(plotter, tag='peak-line')


def plot_edge():
    def _plot(points, tag):
        if len(points) == 0:
            return

        y, x, s = np.transpose(points)
        plotter = PlotterScatter(x, y, s, 6)
        conts_tree_view.append_plot(plotter, tag)

    points = []
    img = np.zeros_like(img_b)

    l_rc = get_rc(0, rr - 3, 0, cc - 3)

    for r0, c0 in l_rc:
        bt = -np.pi / 4
        lp = [[r0 + i, c0 + i] for i in range(3)]
        l_dt = [angle_between_tans(img_t[r, c], bt) for r, c in lp]
        l_p = [img_p[r, c] for r, c in lp]

        s = np.sum([np.cos(dt) * p for dt, p in zip(l_dt, l_p)])
        s -= (img_p[r0 + 1, c0] + img_p[r0, c0 + 1] +
              img_p[r0 + 1, c0 + 2] + img_p[r0 + 2, c0 + 1]) * 0.4

        if s <= -10:
            continue

        r, c = r0 + 1.5, c0 + 1.5
        #img[r, c] = s
        points.append([r, c, s])

    _plot(points, 'peak-edge')


def plot_corner():
    def _is_corner(r0, c0):
        def _is_corner(rot):
            if rot == 0:
                l = [[r0 - 1, c0 - 1], [r0, c0]]
            if rot == 1:
                l = [[r0 - 1, c0], [r0, c0 - 1]]

            p1, p2 = [img_p[r, c] for r, c in l]
            t1, t2 = [img_t[r, c] for r, c in l]

            #print(p1, p2, t1, t2)

            if calc_ratio(p1, p2) < 0.5 or angle_between_tans(t1, t2) < np.pi / 2:
                return

            i1 = quantize_grad_t(t1, 4)[0][0]
            i2 = quantize_grad_t(t2, 4)[0][0]
            #print(i1, i2)

            if rot == 0:
                if (i1 == 2 and i2 == 0) or (i1 == 6 and i2 == 4):
                    return [r0 - 1, c0]
                if (i1 == 0 and i2 == 2) or (i1 == 4 and i2 == 6):
                    return [r0, c0 - 1]
            if rot == 1:
                if (i1 == 2 and i2 == 4) or (i1 == 6 and i2 == 0):
                    return [r0 - 1, c0 - 1]
                if (i1 == 4 and i2 == 2) or (i1 == 0 and i2 == 6):
                    return [r0, c0]

        ret = _is_corner(0)
        if ret:
            return ret
        ret = _is_corner(1)
        if ret:
            return ret

    l_rc = get_rc(1, rr - 1, 1, cc - 1)
    #l_rc = [[4, 4]]

    conts = []
    for r0, c0 in l_rc:
        ret = _is_corner(r0, c0)
        if ret is None:
            continue
        r, c = ret
        r, c = r + 0.5, c + 0.5
        conts.append([[c0, r], [c, r], [c, r0]])

    conts = np.array(conts) - 0.5
    conts_tree_view.append_cont(conts, tag='corner')


def plot_non_maximum_suppression():
    def _is_maximum(r0, c0):
        v_p0, v_t0 = img_p[r0, c0], img_t[r0, c0]

        r1, r2 = max([r0 - 1, 0]), min([r0 + 2, rr - 1])
        c1, c2 = max([c0 - 1, 0]), min([c0 + 2, cc - 1])
        im_p = img_p[r1: r2, c1: c2]
        im_t = img_t[r1: r2, c1: c2]

        if v_p0 < 10:
            return False

        if np.max(im_p) / 4 > v_p0:
            return False

        im_tb = np.zeros((r2 - r1, c2 - c1))
        for i, (r, c) in enumerate(get_8d()):
            r += r0 - r1
            c += c0 - c1

            if r < 0 or c < 0 or r >= r2 - r1 or c >= c2 - c1:
                continue

            im_tb[r, c] = i / 4 * np.pi

        imd_t1 = np.abs(angle_between_tans(im_tb, v_t0)) / np.pi
        imd_t2 = np.abs(angle_between_tans(im_tb, im_t)) / np.pi

        def _eval(im):
            return im < 0.1

        imd_tm = (_eval(imd_t1) & _eval(imd_t2)) | (
            _eval(1 - imd_t1) & _eval(1 - imd_t2))

        # print(imd_t1)
        # print(imd_t2)
        # print(imd_tm)

        l_rc = np.transpose(np.where(imd_tm))
        lp = np.array([im_p[r, c] for r, c in l_rc])

        return not np.any(lp > v_p0 * 2.5)

    l_rc = get_rc(0, rr - 1, 0, cc - 1)
    #l_rc = [[27, 20]]

    points = []

    for r, c in l_rc:
        if _is_maximum(r, c):
            img_edge_mask[r, c] = 1
            points.append([r + 0.5, c + 0.5])

    plotter = PlotterPoints(points)
    conts_tree_view.append_plot(plotter, tag='edge_max')


def plot_grad_edge():
    def _mod(i): return (i + 8) % 8

    def _put_center(ps):
        ps.insert(1, [0, 0])
        return ps

    def _flip_to_align_with_edge(ps):
        # print(ps)

        l_dot = []

        for i in range(len(ps) - 1):
            c1, r1 = ps[i]
            c2, r2 = ps[i + 1]

            dr, dc = r1 - r2, c1 - c2
            dx, dy = dc, -dr
            dx, dy = np.array([dx, dy]) / np.linalg.norm([dx, dy])

            dx1, dy1 = [img_y[r1, c1], -img_x[r1, c1]]
            dx2, dy2 = [img_y[r2, c2], -img_x[r2, c2]]

            #print(dx, dy, dx1, dy1, dx2, dy2)

            dot1 = np.dot([dx1, dy1], [dx, dy])
            dot2 = np.dot([dx2, dy2], [dx, dy])

            l_dot.extend([dot1, dot2])

        # print(l_dot)

        if np.any(np.abs(l_dot) < 0.4):
            return []

        l = np.where(np.abs(l_dot) < 0.6, 1, 0)
        l_li = connect_line_bb(l)
        l_li = [li for li in l_li if len(li) > 1]

        # print(l_li)

        if len(l_li) > 0:
            return []

        if np.mean(l_dot) > 0:
            return ps[::-1]

        return ps

    def _edge(r0, c0):
        if img_edge_mask[r0, c0] == 0:
            return []

        im_mask_edge = img_edge_mask[r0 - 1: r0 + 2, c0 - 1: c0 + 2]

        im_t = img_t[r0 - 1: r0 + 2, c0 - 1: c0 + 2]
        imd_t = angle_between_tans(im_t, im_t[1, 1])
        im_mask_t = np.where(imd_t < 1.3, 1, 0)

        # print(imd_t)
        # print(im_mask_t)

        im_mask = im_mask_t * im_mask_edge

        # print(im_mask)

        sum_mask = np.sum(im_mask)
        if sum_mask <= 1 or sum_mask >= 8:
            return []

        im_p = img_p[r0 - 1: r0 + 2, c0 - 1: c0 + 2]
        im_np = im_p / im_p[1, 1]
        im_mask_p = np.where(im_np > 0.4, 1, 0)

        # print(im_np)
        # print(im_mask_p)

        im_mask = im_mask * im_mask_p

        # print(im_mask)

        l_mask = [im_mask[ro + 1, co + 1] for ro, co in get_8d()]
        l_mask = 1 - np.array(l_mask)

        l_li = connect_line_bb_circular(l_mask)

        # print(l_li)

        l_li = [[_mod(li[0] - 1), _mod(li[-1] + 1)] for li in l_li]
        l_li = [li for li in l_li if _mod(li[1] - li[0]) > 2]

        l_ps = [[get_8d()[i] for i in li] for li in l_li]
        l_ps = [_put_center(ps) for ps in l_ps]

        # print(l_ps)

        l_ps = [[[c + c0, r + r0] for r, c in ps] for ps in l_ps]
        l_ps = [_flip_to_align_with_edge(ps) for ps in l_ps]
        l_ps = [ps for ps in l_ps if len(ps) > 0]

        # print(l_ps)

        return l_ps

    l_rc = get_rc(1, rr - 2, 1, cc - 2)
    #l_rc = [[21, 15]]
    #l_rc = [[7, 8]]

    map_con = ImageMap()

    for r, c in l_rc:
        conts = _edge(r, c)

        if len(conts) == 0:
            continue

        map_con.append(r, c, conts)

    conts_tree_view.append_map_conts(map_con, tag='grad_edge')


def plot_grad_edge2():
    def _edge(r0, c0):
        if img_p[r0, c0] < 3:
            return

        def _step_up(r0, c0, d):
            i, _ = quantize_grad_t(img_t[r0, c0])[0]
            i = i + (0 if d > 0 else 4)
            i = i % 8

            ro, co = get_8d()[i]
            r1, c1 = r0 + ro, c0 + co

            if not box_in(r1, c1, rr - 1, cc - 1):
                return []

            return [r1, c1]

        def _find_edge(r0, c0, d):
            p0 = img_p[r0, c0]

            r1, c1 = r0, c0
            for _ in range(2):
                ret = _step_up(r1, c1, d)

                if len(ret) == 0:
                    return 'none', None

                r, c = ret
                p = img_p[r, c]

                if p / p0 < 1 / 2:
                    return 'found_edge', [r1, c1]

                a = angle_between_tans(img_t[r0, c0], img_t[r, c])

                if a > np.pi / 2:
                    return 'found_edge', [r1, c1]

                if a > 0.5:
                    return 'none', None

                if p > p0:
                    return 'not_maximum', None

                r1, c1 = r, c

            return 'none', None

        edges = []

        for d in [1, -1]:
            result, pos = _find_edge(r0, c0, d)
            if result == 'not_maximum':
                return
            edges.append(pos)

        return edges

    l_rc = get_rc(0, rr - 1, 0, cc - 1)
    # l_rc = [[13, 15]]

    points1, points2 = [], []
    map_1, map_2 = ImageMap(), ImageMap()
    for r, c in l_rc:
        ret = _edge(r, c)
        if not ret:
            continue

        p1, p2 = ret
        if p1:
            points1.append(p1)
            map_1.append(*p1, True)
        if p2:
            points2.append(p2)
            map_2.append(*p2, True)

    plotter = PlotterPoints(np.array(points1) + 0.5)
    conts_tree_view.append_plot(plotter, 'grad-edge-1')
    plotter = PlotterPoints(np.array(points2) + 0.5)
    conts_tree_view.append_plot(plotter, 'grad-edge-2')


def plot_grad_area():
    l_rc = get_rc(0, rr - 1, 0, cc - 1)

    points = []
    for r, c in l_rc:
        t = img_t[r, c]
        if angle_between_tans(t, np.pi * 2 / 4) < np.pi / 4:
            points.append([r + 0.5, c + 0.5])

    conts_tree_view.append_plot(PlotterPoints(points), 'grad-area')


def plot_depression_palm():
    def _is_depressed(r0, c0):
        i = np.argmax([abs(img_b[r, c] - img_b[r0, c0])
                       for r, c in get_8d(r0, c0)])
        r1, c1 = get_8d(r0, c0)[i]
        r2, c2 = get_8d(r0, c0)[(i + 4) % 8]

        lv = np.array([img_b[r1, c1], img_b[r0, c0], img_b[r2, c2]])
        ld = lv[1:] - lv[:-1]
        if not (ld[0] < -2 and ld[1] > 2):
            return False

        # 鞍点回避
        im = img_b[r0 - 1: r0 + 2, c0 - 1: c0 + 2] - img_b[r0, c0]
        im = im / im[r1 - r0 + 1, c1 - c0 + 1]
        return abs(np.min(im)) < 0.4

    l_rc = get_rc(1, rr - 1, 1, cc - 1)
    points = []
    for r0, c0 in l_rc:
        if _is_depressed(r0, c0):
            points.append([r0, c0])

    conts_tree_view.append_plot(PlotterPoints(points), 'depression-palm')


def get_cut_8d(r0=0, c0=0):
    l = [[0, 1], [-.5, .5], [-1, 0], [-.5, -.5],
         [0, -1], [.5, -.5], [1, 0], [.5, .5]]
    return [[r0 + ro, c0 + co] for ro, co in l]


def cut_p2ps(r, c):
    if int(r) == r:
        r0 = r1 = int(r)
        c0 = int(c)
        c1 = c0 + 1
    else:
        c0 = c1 = int(c)
        r0 = int(r)
        r1 = r0 + 1
    return r0, c0, r1, c1


def cut_quantize(r1, c1, r2, c2):
    if int(r1) != r1:
        r = r1
    elif int(r2) != r2:
        r = r2
    else:
        r = (r1 + r2) / 2

    if int(c1) != c1:
        c = c1
    elif int(c2) != c2:
        c = c2
    else:
        c = (c1 + c2) / 2

    return r, c


def cut_min_dif(img, r0, c0, i):
    def _dif(r, c):
        r0, c0, r1, c1 = cut_p2ps(r, c)
        return abs(img[r0, c0] - img[r1, c1])

    d0 = _dif(r0, c0)
    d1 = _dif(*get_cut_8d(r0, c0)[i])
    return min(d0, d1)


def plot_hoge():
    def _plot_map(map: ImageMap, tag, scale=0.5, cl=(1, 0, 0, 0.5)):
        conts = []

        for r0, c0, (li1, li2) in map.get_list():
            l_po = [get_cut_8d()[i] for i in li1 + li2]
            _conts = [[[c0, r0], [c0 + co * scale, r0 + ro * scale]]
                      for ro, co in l_po]
            conts.extend(_conts)

        conts_tree_view.append_plot(Plotter(conts, cl=cl), tag)

    map_edge = ImageMap()

    for r0, c0, cuts in map_cut_2x2.get_list():
        p0 = np.array([r0, c0])
        for cut in cuts:
            ps = np.array(cut.ps)
            ps = ps + p0
            dp = ps[1] - ps[0]
            dp = dp / np.linalg.norm(dp)
            i0 = quantize_grad_t(np.arctan2(-dp[0], dp[1]))[0][0]
            li = [i0, (i0 + 4) % 8]
            for j, (r, c) in enumerate(ps):
                l_li = map_edge.get(r, c)
                if l_li is None:
                    l_li = ([], [])
                de = 1 - j
                l_li[de].append(li[j])
                map_edge.append(r, c, l_li)

    for _, _, l_li in map_edge.get_list():
        for li in l_li:
            li.sort()

    _plot_map(map_edge, 'edge')

    map_grad = ImageMap()

    l_rc = get_rc(0, rr - 1, 0, cc - 1)
    # l_rc = [[6, 1]]

    for r0, c0 in l_rc:
        p0, t0 = img_p[r0, c0], img_t[r0, c0]
        i0 = quantize_grad_t(t0)[0][0]
        i0 = (i0 + 4) % 8
        ro, co = get_8d()[i0]

        for j in [1, 2]:
            r1, c1 = r0 + j * ro, c0 + j * co
            if not box_in(r1, c1, rr - 1, cc - 1):
                break
            p1, t1 = img_p[r1, c1], img_t[r1, c1]
            if angle_between_tans(t0, t1) > np.pi / 3:
                break
            if calc_ratio(p0, p1) < 0.5:
                break
            for r, c in [[r0, c0], [r1, c1]]:
                grad = map_grad.get(r, c)
                if grad is None or j > grad:
                    map_grad.append(r, c, j)

    xs, ys, ss = [], [], []
    for r0, c0, s in map_grad.get_list():
        xs.append(c0 + 0.5)
        ys.append(r0 + 0.5)
        ss.append(s * 3)
    conts_tree_view.append_plot(PlotterScatter(xs, ys, ss, 3), 'grad')

    def _is_peak(r0, c0):
        i = np.argmax([abs(img_b[r, c] - img_b[r0, c0])
                       for r, c in get_8d(r0, c0)])
        r1, c1 = get_8d(r0, c0)[i]
        r2, c2 = get_8d(r0, c0)[(i + 4) % 8]

        lv = np.array([img_b[r1, c1], img_b[r0, c0], img_b[r2, c2]])
        ld = lv[1:] - lv[:-1]
        if ld[0] * ld[1] >= 0:
            return 0, -1

        return min(np.abs(ld)), (i + 2) % 8

    l_rc = get_rc(1, rr - 1, 1, cc - 1)
    # l_rc = [[10, 4]]

    conts = []
    map_line_blob = ImageMap()

    for r0, c0 in l_rc:
        peak, i = _is_peak(r0, c0)
        if peak > 3:
            map_line_blob.append(r0, c0, (peak, i))

            ro, co = get_8d()[i]
            conts.append([
                [c0 - co * 0.5, r0 - ro * 0.5],
                [c0 + co * 0.5, r0 + ro * 0.5]
            ])

    conts = [np.array(cont) - 0.5 for cont in conts]
    conts_tree_view.append_cont(conts, tag='line-blob')

    def _quantize_cut(r1, c1, r2, c2):
        r = round((r1 + r2 - 1) / 2)
        c = round((c1 + c2 - 1) / 2)
        return r, c

    def _find_grad(r0, c0, li, de, d):
        def _check_dif(r0, c0, i0, r1, c1, i1):
            d0 = cut_min_dif(img_b, r0, c0, i0)
            d1 = cut_min_dif(img_b, r1, c1, i1)
            return d0 < d1 * 2

        if len(li) == 0:
            return []

        li_grad = []

        for j in range(len(li)):
            i0 = li[j]
            i1 = li[(j - d) % len(li)]

            di = ((i0 - i1) * d) % 8

            # if di == 1:
            #     # print('di == 1: ', r0, c0, i0, i1)
            #     if _check_dif(r0, c0, i0, r0, c0, i1):
            #         li_grad.append(i0)
            #     continue
            if di == 2:  # ピクセル外にはないでしょ
                continue

            r, c = _quantize_cut(r0, c0, *get_cut_8d(r0, c0)[i0])
            grad = map_grad.get(r, c)
            if grad and grad >= 2:
                li_grad.append(i0)
                continue

            # ピクセル外処理
            for di in [1, 2, 3]:
                ro, co = get_cut_8d()[(i0 - d * di) % 8]
                r1, c1 = r0 + ro, c0 + co

                l_li1 = map_edge.get(r1, c1)
                if l_li1 is None:
                    continue

                li1 = l_li1[de]

                # if any([chaincode_dif_abs(i0, i1) <= 1 for i1 in li1]):
                found = False
                for i1 in li1:
                    if not (i0 == i1 or (di != 3 and i0 == (i1 - d) % 8)):
                        continue

                    if not _check_dif(r0, c0, i0, r1, c1, i1):
                        continue

                    li_grad.append(i0)

                    # print((r0, c0, i0), (r1, c1, i1))

                    found = True
                    break

                if found:
                    break

        return li_grad

    de = 1
    d = 1

    l_edge = map_edge.get_list()
    # l_edge = [[r, c, map_edge.get(r, c)] for r, c in (
    #     [[2, 2.5]]
    # )]

    map_edge_no_grad = ImageMap()

    for r0, c0, l_li in l_edge:
        li = l_li[de]
        li_grad = _find_grad(r0, c0, li, de, d)
        li_no_grad = [i for i in li if i not in li_grad]
        if not li_no_grad:
            continue

        l_li_no_grad = map_edge_no_grad.get(r0, c0)
        if l_li_no_grad is None:
            l_li_no_grad = ([], [])

        l_li_no_grad[de].extend(li_no_grad)
        map_edge_no_grad.append(r0, c0, l_li_no_grad)

        for i in li_no_grad:
            r1, c1 = get_cut_8d(r0, c0)[i]
            l_li2 = map_edge_no_grad.get(r1, c1)
            if l_li2 is None:
                l_li2 = ([], [])

            l_li2[1 - de].append((i + 4) % 8)
            map_edge_no_grad.append(r1, c1, l_li2)

    _plot_map(map_edge_no_grad, 'no-grad', cl='dodgerblue')

    class TraceResult:
        def __init__(self):
            self.cont = []
            self.cont_sub_blob = []
            # sub_blobにmain_contが刺さって止まっている場合
            self.prefer_sub_blob = False
            self.count_grad = 0

    def _trace_one(result: TraceResult, de, N, branch, debug=True):
        if result.count_grad > 3:
            if debug:
                print('count_grad > 5')
            return 'break', None

        c0, r0 = result.cont[-1]
        l_li = map_edge_no_grad.get(r0, c0)
        if l_li and l_li[de]:
            result.count_grad = 0
        else:
            result.count_grad += 1

        l_li = map_edge.get(r0, c0)
        if not (l_li and l_li[de]):
            if debug:
                print('map_edge is None')
            return 'break', None

        li = l_li[de]
        if len(li) == 1:
            i = li[0]
            r0, c0 = get_cut_8d(r0, c0)[i]
            result.cont.append([c0, r0])
            return 'continue', None

        results = []
        for i in li:
            r1, c1 = get_cut_8d(r0, c0)[i]
            branch_result = TraceResult()
            if branch:
                branch_result.cont = result.cont + [[c1, r1]]
            else:
                branch_result.cont = [[c1, r1]]
            branch_result.count_grad = result.count_grad
            results.extend(_trace_all(branch_result, de, N))

        return 'branch', results

    def _trace_all(result: TraceResult, de, N):
        while len(result.cont) < N:
            control, results = _trace_one(result, de, N, True, debug=False)
            if control == 'break':
                break
            elif control == 'continue':
                continue

            assert(control == 'branch')
            return results

        return [result]

    def _cont_contrast(cont):
        lv = []
        for c, r in cont:
            r0, c0, r1, c1 = cut_p2ps(r, c)
            v = abs(img_b[r0, c0] - img_b[r1, c1])

            # LineBlobのピーク分を引く
            # if line_blob:
            #     peak1 = map_line_blob.get(r0, c0)
            #     peak2 = map_line_blob.get(r1, c1)
            #     peak1 = peak1[0] if peak1 else 0
            #     peak2 = peak2[0] if peak2 else 0
            #     peak = max(peak1, peak2)
            #     v = max(v - peak, 0)
            lv.append(v)
        return lv

    def _find_sub_blob(result):
        cont = result.cont
        if len(cont) > 1 and cont[0] == cont[-1]:
            return False

        p0 = cont[-1]
        for i in range(len(cont) - 2, 0, -1):
            if p0 == cont[i]:
                result.cont_sub_blob = cont[i:]
                result.cont = cont[:i + 1]
                return True

        print('error at _find_sub_blob', cont)
        return False

    def _trace(cont, de, current_index):
        def _finish():
            c, r = result.cont[-1]
            index = map_done.get(r, c)
            if index is None:
                return False
            if index != current_index:
                return True  # 他の輪郭に当たった
            if sub_blob_count > 2:
                result.cont = result.cont[:-3]
                result.prefer_sub_blob = True
                return True  # サブブロブを切り離して、メインを伸ばしたが結局通常のブロブ
            if sub_blob_count > 0:
                return False  # サブブロブを切り離して、メインを伸ばし中
            # print(r, c)
            # for r1, c1, j in map_done.get_list():
            #     if j == index:
            #         print(r1, c1)
            # print()
            if not _find_sub_blob(result):
                return True  # 通常のブロブ
            return False

        result = TraceResult()
        result.cont = cont

        sub_blob_count = 0

        for _ in range(10000000):
            if _ > 0 and _finish():
                break

            if result.cont_sub_blob:
                sub_blob_count += 1
            else:
                sub_blob_count = 0

            c, r = result.cont[-1]
            map_done.append(r, c, current_index)

            control, results = _trace_one(result, de, 3, False, debug=False)
            if control == 'break':
                break
            elif control == 'continue':
                continue

            assert(control == 'branch')

            len_max = max([len(res.cont) for res in results])
            results = list(filter(lambda result: len(
                result.cont) == len_max, results))

            # print(r, c)
            l_score = []
            for _result in results:
                if _result.cont[-1] in result.cont[-3:]:
                    v = 0  # ブロブ
                else:
                    v = np.mean(_cont_contrast(_result.cont))

                l_score.append(v)
                # print(_result.cont, v, result.cont[-1])
            # print()

            j = np.argmax(l_score)
            result.cont.append(results[j].cont[0])

        if result.count_grad > 1:
            end = 1 - result.count_grad
            for j in range(end, 0):
                c, r = cont[j]
                map_done.remove(r, c)
            result.cont = result.cont[:end]

        return result.cont, result.cont_sub_blob, result.prefer_sub_blob

    def _switch_branch(cont0, de, current_index):
        def _score_curvature(cont):
            l_ti = []
            for j1, j2 in range_adj(len(cont)):
                c1, r1 = cont[j1]
                c2, r2 = cont[j2]
                i = get_cut_8d().index([r2 - r1, c2 - c1])
                l_ti.append(i)

            dt = chaincode_dif_abs(l_ti[0], l_ti[-1])
            # (dt, score) = (0, 1), (1, 1), (2, 0.5), (3, 0), (4, 0)
            return max(min((3 - dt) / 2, 1), 0)

        def _score(cont):
            lv = _cont_contrast(cont[1:])
            # print(lv)
            return np.sum(lv)

        c, r = cont0[-1]
        i_cont = map_done.get(r, c)
        if i_cont is None or i_cont == current_index or i_cont in blob_indices:
            return 'NOT_BRANCH', cont0

        cont1 = conts[i_cont]

        try:
            i = cont1.index([c, r])
        except Exception as e:
            print(e)
            return 'NOT_BRANCH', cont0

        # print(de, cont0)
        # print(i, cont1)
        # print()

        if de == 0 and i == len(cont1) - 1:
            contrast1 = np.mean(_cont_contrast(cont0[-5:-1]))
            contrast2 = np.mean(_cont_contrast(cont1[-1:-5:-1]))
            if calc_ratio(contrast1, contrast2) > 0.5:
                cont = cont0 + cont1[::-1][1:]
                del cont1[:]
                return 'EXTENDED', cont
            else:
                return 'NOT_BRANCH', cont0
        if de == 1 and i == 0:
            contrast1 = np.mean(_cont_contrast(cont0[-5:-1]))
            contrast2 = np.mean(_cont_contrast(cont1[:4]))
            if calc_ratio(contrast1, contrast2) > 0.5:
                cont = cont0 + cont1[1:]
                del cont1[:]
                return 'EXTENDED', cont
            else:
                return 'NOT_BRANCH', cont0

        sample_cont0 = cont0[::-1][:4]

        if len(sample_cont0) < 4:
            return 'BRANCH_BUT_NO_CHANGE', cont0

        if de == 0:
            sample_cont1 = cont1[i:][:4]
        else:
            sample_cont1 = cont1[:i + 1][::-1][:4]

        sample_conts = [sample_cont0, sample_cont1]

        ls1 = np.array([_score(cont) for cont in sample_conts])
        max_score = max(ls1)
        if max_score > 0:
            ls1 = ls1 / max_score

        ls2 = np.array([_score_curvature(cont) for cont in sample_conts])

        score0, score1 = ls1 + ls2 * 0.7

        # print(sample_cont0, sample_cont1)
        # print(ls1, ls2)
        # print(score0, score1)
        # print()

        if score0 > score1:
            if de == 0:
                cont = cont0 + cont1[:i][::-1]
                del cont1[:i]
                return 'BRANCH_OVERRIDE', cont
            else:
                cont = cont0 + cont1[i + 1:]
                del cont1[i + 1:]
                return 'BRANCH_OVERRIDE', cont
        return 'BRANCH_BUT_NO_CHANGE', cont0

    def _switch_check_pattern(r0, c0, cut_i):
        l_po = [[co * 0.5, ro * 0.5] for ro, co in get_8d()[::2]]
        l_p = [[c0 + 0.5 + co, r0 + 0.5 + ro] for co, ro in l_po]
        li_cont = [map_done.get(r, c) for c, r in l_p]

        if any([i is None for i in li_cont]):
            return

        if len(set(li_cont)) != 2:
            return

        error = [False]

        def _cont_sample(i1, i2):
            cont1 = conts[li_cont[i1]]
            try:
                j1 = cont1.index(l_p[i1])
            except:
                print('error', cont1, l_p[i1])
                error[0] = True
                return [], [], -1
            j2 = j1 + 1
            if cont1[j2] != l_p[i2]:
                print('error')
                error[0] = True
                return [], [], j1
            N = 5
            j1_b, j2_e = max(j1 - N, 0), min(j2 + N + 1, len(cont1))
            return cont1[j1_b: j1 + 1], cont1[j2: j2_e], j1

        def _score_curvature(cont):
            l_ti = []
            for j1, j2 in range_adj(len(cont)):
                c1, r1 = cont[j1]
                c2, r2 = cont[j2]
                i = get_cut_8d().index([r2 - r1, c2 - c1])
                l_ti.append(i)

            dt = chaincode_dif_abs(l_ti[0], l_ti[-1])
            # (dt, score) = (0, 1), (1, 1), (2, 0.5), (3, 0), (4, 0)
            return max(min((3 - dt) / 2, 1), 0)

        def _score(conts, lv):
            sv = (calc_ratio(lv[0], lv[1]) + calc_ratio(lv[2], lv[3])) / 2
            sc = (_score_curvature(conts[0][-2:] + conts[1][:2]) +
                  _score_curvature(conts[2][-2:] + conts[3][:2])) / 2
            return (sv + sc) / 2

        # cut.i == 6
        # 0 1
        # 1 0

        # cut.i == 9
        # 1 0
        # 0 1

        if li_cont[0] == li_cont[1] and li_cont[2] == li_cont[3]:
            if cut_i == 6:
                li = [3, 2, 1, 0]
            else:
                li = [0, 1, 2, 3]
        elif li_cont[1] == li_cont[2] and li_cont[3] == li_cont[0]:
            if cut_i == 6:
                li = [1, 2, 3, 0]
            else:
                li = [2, 1, 0, 3]
        else:
            return

        li2 = li[::-1]
        li2 = li2[1:] + li2[:1]

        conts_sample = [None for _ in range(4)]
        conts_sample[li[0]], conts_sample[li[1]
                                          ], j1 = _cont_sample(li[0], li[1])
        conts_sample[li[2]], conts_sample[li[3]
                                          ], j2 = _cont_sample(li[2], li[3])

        if error[0]:
            return

        lv = [np.mean(_cont_contrast(cont)) for cont in conts_sample]

        def _arg(li): return ([conts_sample[i]
                               for i in li], [lv[i] for i in li])

        s1 = _score(*_arg(li))
        s2 = _score(*_arg(li2))
        if s1 >= s2:
            return

        # print(s1, s2)

        i_cont1, i_cont2 = [li_cont[li[i]] for i in [0, 2]]

        new_cont1 = conts[i_cont2][j2 + 1:]
        new_cont2 = conts[i_cont1][j1 + 1:]

        cont1 = conts[i_cont1][:j1 + 1] + new_cont1
        cont2 = conts[i_cont2][:j2 + 1] + new_cont2

        conts[i_cont1] = cont1
        conts[i_cont2] = cont2

        for i, (c, r) in enumerate(new_cont1):
            if i == len(new_cont1) - 1:
                if map_done.get(r, c) != i_cont2:
                    continue
            map_done.append(r, c, i_cont1)
        for i, (c, r) in enumerate(new_cont2):
            if i == len(new_cont2) - 1:
                if map_done.get(r, c) != i_cont1:
                    continue
            map_done.append(r, c, i_cont2)

    def _append_blob(blob):
        for c, r in blob:
            map_done.append(r, c, len(conts))
        if blob:
            blob_indices.append(len(conts))
            conts.append(blob)

    l_rc = [[r, c] for r, c, _ in map_edge_no_grad.get_list()]

    map_done = ImageMap()
    conts = []
    blob_indices = []
    max_count = 44

    if False:
        max_count = 10
        l_rc = [[12.5, 14]]
        conts = [[[0.0, 11.5], [1.0, 11.5], [1.5, 11.0], [1.0, 10.5], [0.5, 10], [0.5, 9], [0.5, 8], [0.5, 7], [0.5, 6], [0.5, 5], [0.5, 4], [0.5, 3], [0.5, 2], [0.5, 1], [0.5, 0]], [[2.0, 0.5], [1.5, 0]], [[2.5, 12.0], [2.5, 13.0], [3.0, 13.5], [3.5, 13.0], [3.5, 12.0], [3.0, 11.5], [2.5, 12.0]], [[3.5, 0], [3.0, 0.5], [2.5, 1.0], [2.5, 2.0]], [[5.5, 1.0], [6.0, 0.5], [6.5, 1.0], [6.0, 1.5], [5.5, 1.0]], [[5.5, 1.0], [5.0, 1.5], [4.5, 1], [4.5, 0]], [[5.5, 0], [5.5, 1]], [[5.5, 2.0], [6.0, 2.5], [6.5, 2.0], [7.0, 1.5], [7.5, 1], [7.5, 0]], [[8.5, 1], [8.0, 1.5], [7.5, 1.0]], [[12.0, 1.5], [13.0, 1.5], [13.5, 1.0], [13.0, 0.5], [12.0, 0.5], [11.5, 1.0], [12.0, 1.5]], [[11.5, 0], [11.0, 0.5], [10.5, 1.0], [11.0, 1.5], [12.0, 1.5]], [], [[19.5, 3.0], [20.0, 3.5], [20.5, 3.0], [20.5, 2.0], [20.0, 1.5], [19.5, 2.0], [19.5, 3.0]], [[19.5, 3.0], [19.0, 2.5], [18.5, 2.0], [18.5, 1.0], [18.0, 0.5], [17.5, 0]], [[5.5, 1.0], [5.5, 2.0], [5.5, 3.0], [5.5, 4.0], [5.5, 5.0], [5.5, 6.0], [5.0, 6.5], [4.5, 6.0], [4.5, 5.0], [4.5, 4.0], [4.5, 3.0], [4.5, 2.0], [4.0, 1.5], [3.5, 1.0], [4, 0.5], [4.5, 0.0]], [[4.5, 1.0], [5, 0.5], [5.5, 1.0]], [[7.5, 1.0], [8, 0.5], [8.5, 1.0]], [[21.5, 1.0], [22.0, 1.5], [22.5, 1.0], [22, 0.5], [21.5, 1.0]], [], [], [[18.5, 4.0], [18.0, 3.5], [17.5, 3.0], [17.0, 2.5], [16.5, 2.0], [16.0, 1.5], [15.0, 1.5], [14.5, 1], [14.0, 0.5], [13.5, 0.0]], [], [], [[2.5, 3.0], [2.0, 3.5], [1.5, 3], [1.5, 2], [2.0, 1.5]], [[4.0, 1.5], [3.5, 2], [4.0, 2.5], [4.5, 2.0]], [[8.5, 0], [8.5, 1], [8.5, 2], [8.0, 2.5], [7.5, 3.0], [7.0, 3.5], [6.5, 3.0]], [[8.5, 0.0], [9.0, 0.5], [9.5, 1], [9.5, 2], [9.5, 3], [9.5, 4], [9.0, 4.5], [8.5, 5.0], [8.0, 5.5], [7.5, 6.0], [7.0, 6.5], [6.5, 7.0], [6.5, 8.0], [6.5, 9.0], [7.0, 9.5], [8.0, 9.5]], [[20.5, 3.0], [21.0, 3.5], [21.5, 4.0], [22.0, 4.5], [22.5, 4.0], [23.0, 3.5], [23.5, 3.0], [24.0, 2.5], [24.5, 2.0], [25, 1.5], [26, 1.5], [27, 1.5]], [[6.5, 3.0], [7, 2.5], [7.5, 3.0]], [[9.5, 2.0], [10, 2.5], [11, 2.5], [12, 2.5], [13, 2.5], [13.5, 3.0], [13.5, 4.0], [13.5, 5.0], [13.0, 5.5], [12.0, 5.5], [11.5, 5.0], [11.0, 4.5], [10.0, 4.5], [9.0, 4.5]], [[22.0, 0.5], [21.5, 1.0], [21.0, 1.5], [20.5, 2.0], [20, 2.5], [19.5, 2.0], [19.5, 1], [20.0, 0.5], [20.5, 1.0], [20.5, 2.0]], [[21.5, 2.0], [22, 2.5], [23, 2.5], [24, 2.5]], [[13.5, 3.0], [13, 3.5], [12, 3.5], [11, 3.5], [10.5, 4.0], [10.0, 4.5]], [], [[23.0, 3.5], [23.5, 4.0], [24.0, 4.5], [24.5, 4.0], [25, 3.5], [25.5, 4.0], [25.5, 5.0], [25.5, 6.0], [25.5, 7.0], [25.5, 8.0], [26.0, 8.5], [26.5, 9.0], [26.5, 10.0], [26.5, 11.0], [26.5, 12.0], [27.0, 12.5]], [[5.5, 4.0], [6.0, 4.5], [6.5, 4], [6.5, 3], [6.5, 2.0]], [[7.5, 3], [7.5, 4], [7.0, 4.5], [6.5, 4.0]], [[23.0, 16.5], [23.5, 17.0], [24.0, 17.5], [24.5, 18.0], [25.0, 18.5], [25.5, 19.0], [26.0, 19.5], [26.5, 19.0], [26.5, 18.0], [26.5, 17.0], [26.0, 16.5], [25.5, 16.0], [25.0, 15.5], [24.5, 15.0], [24.0, 14.5], [23.0, 14.5], [22.5, 14.0], [22.0, 13.5], [21.0, 13.5], [20.5, 13.0], [20.0, 12.5], [19.0, 12.5], [18.5, 12.0], [19.0, 11.5], [19.5, 11], [19.5, 10], [19.5, 9], [19.5, 8], [19.5, 7], [19.5, 6], [19.5, 5], [19.5, 4], [19.0, 3.5], [18.5, 3.0], [18.0, 2.5], [17.5, 2.0], [17.0, 1.5], [16.5, 1.0], [16.0, 0.5], [15.5, 0]], [[21.5, 1], [21.5, 2], [21.0, 2.5], [20.5, 3], [20.5, 4], [20.5, 5], [20.5, 6], [20.5, 7], [20.5, 8], [20.5, 9], [20.5, 10], [20.5, 11], [20.5, 12], [20.0, 12.5]], [[4.5, 6.0], [4.0, 5.5], [3.5, 5], [4.0, 4.5], [4.5, 4.0]], [[6.0, 4.5], [6.5, 5], [6.5, 6], [6.0, 6.5], [5.5, 7.0], [5.0, 7.5], [4.5, 8.0], [4.5, 9.0], [5.0, 9.5], [6.0, 9.5], [6.5, 10.0], [7.0, 10.5], [8.0, 10.5], [9.0, 10.5], [10.0, 10.5]], [[13.0, 5.5], [12.5, 5]], [[14.5, 5], [14.0, 5.5], [13.0, 5.5]], [[0.5, 6.0], [0, 5.5]], [[0.5, 6.0], [1, 5.5], [1.5, 6.0], [1.0, 6.5], [0.5, 6.0]], [[4.5, 6.0], [5, 5.5], [5.5, 6.0]], [[15.0, 4.5], [14.5, 5.0], [15, 5.5], [15.5, 6.0], [15.5, 7.0], [15.5, 8.0], [16.0, 8.5], [17.0, 8.5], [17.5, 8.0], [17.5, 7.0], [17.5, 6.0], [17.5, 5.0], [17.5, 4.0], [17, 3.5], [16.5, 4.0]], [[16, 5.5], [15.5, 6.0]], [[17.5, 6.0], [17, 5.5]], [[11.0, 8.5], [11.5, 8.0], [11.5, 7.0], [11.5, 6.0], [11.0, 5.5], [10.0, 5.5], [9.0, 5.5], [8.5, 6], [8.0, 6.5], [7.5, 7.0], [7.5, 8.0], [7.5, 9.0], [8.0, 9.5], [9.0, 9.5], [10.0, 9.5], [10.5, 9.0], [11.0, 8.5], [12.0, 8.5], [13.0, 8.5], [13.5, 9.0], [14.0, 9.5], [15.0, 9.5]], [[11.0, 5.5], [10.5, 6], [10.0, 6.5], [9.0, 6.5]], [[1.5, 6.0], [2, 6.5], [2.5, 7.0]], [[4.0, 8.5], [3.5, 8.0], [3.5, 7.0], [4, 6.5], [4.5, 7.0], [4.5, 8.0]], [[10.5, 6.0], [11, 6.5]], [[11.5, 8.0], [12.0, 7.5], [12.5, 7.0], [12, 6.5], [11.5, 6.0]], [[12.5, 7.0], [13, 6.5], [13.5, 7.0], [13.0, 7.5], [12.5, 7.0]], [[15.5, 6.0], [15, 6.5], [14.5, 7.0], [14.5, 8.0], [15.0, 8.5], [16.0, 8.5]], [[19.5, 7.0], [20, 6.5], [20.5, 7.0]], [[8.0, 9.5], [8.5, 9.0], [9.0, 8.5], [9.5, 8], [9.5, 7], [9.0, 6.5], [8.5, 7.0], [8.0, 7.5], [7.5, 8.0]], [[11.5, 8.0], [11.0, 7.5], [10.5, 7], [11.0, 6.5], [11.5, 6.0]], [[19.5, 8.0], [19.0, 7.5], [18.5, 7], [18.5, 6], [18.5, 5], [18.5, 4], [18.5, 3.0]], [[0.5, 9.0], [1.0, 8.5], [1.5, 8.0], [1, 7.5], [0.5, 7.0]], [[1.5, 8.0], [2, 7.5], [2.5, 8.0]], [[3.5, 8.0], [3, 7.5], [2.5, 8.0]], [[9.5, 8.0], [10, 7.5], [10.5, 7.0]], [[22.5, 8], [23.0, 7.5], [24.0, 7.5], [24.5, 8.0], [24.5, 9.0], [24.0, 9.5], [23.0, 9.5], [22.5, 9.0], [22.5, 8.0]], [[20.5, 10.0], [21.0, 10.5], [21.5, 10.0], [21.5, 9.0], [22.0, 8.5], [22.5, 8]], [[0.5, 9.0], [0, 8.5]], [[2.5, 8.0], [2, 8.5], [1.5, 8.0]], [[22.5, 9.0], [23, 8.5], [23.5, 9.0], [23.0, 9.5]], [[1.0, 8.5], [1.5, 9], [1.5, 10], [1.0, 10.5]], [[4.5, 8.0], [4.0, 8.5], [3.5, 9], [3.5, 10], [3.0, 10.5], [2.5, 11.0]], [[2.5, 9.0], [3, 9.5], [3.5, 10.0]], [[24.5, 9], [24.5, 10], [24.0, 10.5], [23.5, 11.0], [23.5, 12.0]], [[1.5, 11.0], [2, 10.5], [2.5, 11.0]], [[0.0, 11.5], [0.5, 11], [0.5, 10]], [[3.5, 10], [3.5, 11], [3.0, 11.5]], [[21.0, 11.5], [21.5, 11]], [[24.5, 10], [24.5, 11], [24.5, 12], [25.0, 12.5], [26.0, 12.5], [27.0, 12.5]], [[1, 11.5], [2, 11.5], [2.5, 12.0]], [[3.5, 12.0], [4, 11.5], [4.5, 11.0], [5.0, 10.5], [6.0, 10.5], [7.0, 10.5]], [[12.0, 10.5], [13.0, 10.5], [13.5, 11.0], [14, 11.5], [15, 11.5], [16, 11.5], [17, 11.5], [18, 11.5], [18.5, 11.0], [19.0, 10.5], [19.5, 10.0]], [[20.0, 12.5], [19.5, 12.0], [20, 11.5], [20.5, 12.0]], [[22, 11.5], [23, 11.5], [23.5, 12.0], [23.5, 13.0]], [[20.5, 11.0], [21, 11.5], [22.0, 11.5], [22.5, 12], [23.0, 12.5], [23.5, 13.0], [23.0, 13.5], [22.0, 13.5]], [[0.5, 24.0], [1.0, 24.5], [1.5, 24.0], [1.0, 23.5], [0.5, 24.0]], [[0.5, 24.0], [0.5, 23.0], [0.5, 22.0], [0.5, 21.0], [0.5, 20.0], [0.5, 19.0], [0.5, 18.0], [0.5, 17.0], [0.5, 16.0], [0.5, 15.0], [0.5, 14.0], [0.5, 13.0], [0, 12.5]], [[2.5, 0.0], [2.0, 0.5], [1.5, 1], [2.0, 1.5], [2.5, 2.0], [2.5, 3.0], [2.5, 4.0], [2.5, 5.0], [2.5, 6.0], [2.5, 7.0], [2.5, 8.0], [2.5, 9.0], [2.5, 10.0], [2.5, 11.0], [2.5, 12.0], [2, 12.5], [1.5, 13.0], [1.0, 13.5], [0.5, 13.0]]]
        map_done.dict = {'0,0.5': 0, '1,0.5': 0, '2,0.5': 0, '3,0.5': 0, '4,0.5': 0, '5,0.5': 0, '6,0.5': 0, '7,0.5': 0, '8,0.5': 0, '9,0.5': 0, '10,0.5': 0, '10.5,1': 0, '11,1.5': 0, '11.5,1': 0, '11.5,0': 0, '0,1.5': 1, '0.5,2': 87, '0,2.5': 87, '0,3.5': 3, '0.5,3': 3, '1,2.5': 3, '2,2.5': 87, '3,2.5': 87, '4,2.5': 87, '5,2.5': 87, '6,2.5': 87, '7,2.5': 87, '8,2.5': 87, '9,2.5': 87, '10,2.5': 87, '11,2.5': 87, '12,2.5': 87, '13,2.5': 2, '13.5,3': 2, '13,3.5': 2, '12,3.5': 80, '11.5,3': 76, '0,4.5': 5, '1,4.5': 5, '1.5,5': 5, '1,5.5': 15, '0.5,6': 4, '1,6.5': 4, '1.5,6': 4, '0,5.5': 6, '0,7.5': 7, '1,7.5': 7, '1.5,7': 7, '2,6.5': 35, '2.5,6': 7, '2,5.5': 14, '0,8.5': 25, '1,8.5': 25, '1.5,8': 8, '0,11.5': 10, '0.5,11': 10, '1,10.5': 10, '1.5,11': 10, '1.5,12': 9, '1.5,13': 9, '1,13.5': 9, '0.5,13': 9, '0.5,12': 9, '1,11.5': 9, '0,15.5': 37, '0.5,16': 37, '1,16.5': 37, '1.5,17': 37, '2,17.5': 37, '2.5,18': 37, '3,18.5': 60, '4,18.5': 60, '0,17.5': 13, '0.5,18': 13, '1,18.5': 13, '2,18.5': 13, '2.5,19': 13, '3,19.5': 12, '3.5,20': 12, '3,20.5': 38, '2,20.5': 30, '1.5,20': 12, '2,19.5': 30, '0.5,4': 14, '1,3.5': 14, '1.5,4': 14, '2,4.5': 14, '3,4.5': 14, '4,4.5': 14, '5,4.5': 14, '6,4.5': 14, '6.5,5': 14, '6,5.5': 14, '5,5.5': 14, '4,5.5': 14, '3,5.5': 14, '0.5,5': 15, '0.5,8': 16, '0.5,22': 30, '1,22.5': 17, '1.5,22': 17, '1,21.5': 38, '1.5,21': 30, '1,1.5': 87, '1.5,2': 87, '1,14.5': 20, '1.5,15': 20, '1.5,16': 20, '2,16.5': 20, '2.5,17': 20, '3,17.5': 20, '3.5,18': 20, '0.5,14': 20, '0,13.5': 20, '1,19.5': 30, '0.5,20': 30, '1,20.5': 30, '1.5,26': 27, '1.5,25': 27, '2,24.5': 27, '2.5,24': 27, '3,23.5': 27, '3.5,23': 27, '4,22.5': 27, '4.5,22': 27, '4,21.5': 27, '3.5,21': 27, '1.5,27': 27, '2,1.5': 23, '3,1.5': 23, '3.5,2': 23, '2,3.5': 24, '2.5,4': 24, '2,8.5': 25, '2.5,8': 25, '3,7.5': 25, '3.5,7': 25, '3,6.5': 35, '2,9.5': 26, '1,9.5': 26, '0.5,9': 26, '3,9.5': 26, '4,9.5': 26, '4.5,9': 26, '5,8.5': 26, '5.5,8': 26, '6,7.5': 26, '6.5,7': 26, '7,6.5': 26, '8,6.5': 26, '9,6.5': 26, '9.5,7': 26, '9.5,8': 49, '9.5,9': 49, '9.5,10': 49, '9,10.5': 49, '8.5,11': 49, '8.5,12': 49, '8.5,13': 49, '9,13.5': 49, '9.5,14': 49, '9.5,15': 49, '2,21.5': 38, '2.5,21': 38, '2.5,7': 28, '2.5,10': 29, '2.5,11': 29, '2.5,12': 29, '2.5,13': 29, '3,13.5': 29, '4,13.5': 29, '5,13.5': 29, '5.5,13': 29, '5.5,12': 29, '5,11.5': 29, '4.5,11': 29, '4.5,10': 29, '2.5,20': 30, '2.5,22': 31, '2.5,23': 31, '3.5,11': 32, '3.5,12': 32, '3.5,13': 32, '4,10.5': 32, '3.5,17': 46, '4,17.5': 46, '5,17.5': 46, '6,17.5': 46, '7,17.5': 46, '8,17.5': 46, '8.5,17': 46, '4,16.5': 46, '3.5,25': 34, '4,24.5': 34, '4.5,24': 34, '4,23.5': 34, '4,25.5': 34, '5,25.5': 34, '6,25.5': 34, '7,25.5': 34, '8,25.5': 34, '8.5,26': 34, '9,26.5': 34, '10,26.5': 34, '11,26.5': 34, '12,26.5': 34, '12.5,27': 34, '4,6.5': 35, '4.5,6': 35, '4,7.5': 36, '4.5,7': 36, '4,19.5': 37, '5,19.5': 37, '6,19.5': 37, '7,19.5': 37, '8,19.5': 37, '9,19.5': 37, '10,19.5': 37, '11,19.5': 37, '11.5,19': 37, '12,18.5': 37, '12.5,19': 37, '12.5,20': 37, '13,20.5': 37, '13.5,21': 37, '13.5,22': 84, '14,22.5': 37, '14.5,23': 37, '14.5,24': 37, '15,24.5': 37, '15.5,25': 37, '16,25.5': 37, '16.5,26': 37, '17,26.5': 37, '18,26.5': 37, '19,26.5': 37, '19.5,26': 37, '19,25.5': 37, '18.5,25': 37, '18,24.5': 37, '17.5,24': 37, '17,23.5': 37, '16.5,23': 37, '3.5,19': 37, '4,20.5': 38, '5,20.5': 38, '6,20.5': 38, '7,20.5': 38, '8,20.5': 38, '9,20.5': 38, '10,20.5': 38, '11,20.5': 84, '12,20.5': 38, '5,3.5': 39, '5.5,4': 39, '4.5,4': 39, '5,6.5': 40, '6,6.5': 40, '6.5,6': 40, '7,5.5': 40, '7.5,5': 40, '8,4.5': 71, '9,4.5': 40, '9.5,5': 40, '9.5,6': 40, '10,6.5': 40, '10.5,7': 40, '10.5,8': 40, '10.5,9': 40, '10.5,10': 40, '5,12.5': 41, '5,14.5': 46, '4.5,15': 46, '5.5,14': 42, '5.5,0': 43, '5.5,1': 44, '6,1.5': 44, '6.5,1': 44, '5.5,5': 45, '5.5,15': 46, '6,15.5': 46, '7,15.5': 46, '8,15.5': 46, '8.5,16': 46, '5.5,16': 47, '5.5,17': 48, '6,8.5': 49, '5.5,9': 49, '5.5,10': 49, '5.5,11': 49, '6,11.5': 59, '7,11.5': 49, '8,11.5': 49, '6.5,8': 49, '7,7.5': 49, '8,7.5': 58, '9,7.5': 49, '6,10.5': 50, '6.5,10': 50, '6.5,9': 58, '7,8.5': 58, '7.5,8': 58, '6.5,2': 51, '6.5,4': 52, '7,3.5': 52, '8,3.5': 52, '8.5,4': 71, '7,4.5': 52, '6.5,11': 59, '6.5,12': 54, '7,12.5': 54, '7.5,12': 54, '6.5,13': 55, '7,13.5': 55, '7.5,13': 55, '6.5,15': 56, '7,14.5': 56, '8,14.5': 56, '8.5,15': 56, '6.5,20': 57, '7,9.5': 58, '8,9.5': 58, '8.5,9': 58, '9,8.5': 58, '7,10.5': 59, '7.5,11': 59, '7,18.5': 60, '7.5,19': 60, '6,18.5': 60, '5,18.5': 60, '7.5,1': 61, '8,1.5': 61, '8.5,1': 61, '7.5,2': 62, '7.5,3': 63, '7.5,10': 64, '8,22.5': 65, '8.5,22': 66, '9,21.5': 66, '10,21.5': 66, '10.5,21': 66, '7.5,23': 65, '7.5,24': 65, '8,24.5': 65, '9,24.5': 73, '9.5,24': 65, '9.5,23': 69, '9,22.5': 69, '8.5,0': 67, '8.5,2': 68, '8.5,23': 69, '9,23.5': 69, '9,1.5': 70, '10,1.5': 70, '9,3.5': 71, '10,3.5': 71, '10.5,3': 71, '9.5,3': 72, '10,24.5': 73, '10.5,24': 73, '11,23.5': 73, '12,23.5': 83, '13,23.5': 84, '13.5,23': 84, '10.5,2': 74, '11,0.5': 75, '11,3.5': 76, '11,21.5': 77, '11.5,21': 84, '11,24.5': 78, '12,24.5': 78, '12.5,25': 78, '12.5,26': 78, '11.5,2': 79, '11.5,4': 80, '11,4.5': 80, '10.5,5': 80, '10.5,6': 80, '11.5,17': 81, '11.5,16': 81, '11.5,15': 81, '11.5,14': 81, '11,13.5': 81, '10.5,13': 81, '10.5,12': 81, '11.5,18': 81, '11,18.5': 81, '10.5,19': 81, '11.5,20': 82, '12,19.5': 82, '11.5,22': 84, '11.5,23': 83, '12,22.5': 84, '12.5,23': 84, '12.5,0': 86, '13,0.5': 86, '14,0.5': 86, '15,0.5': 86, '16,0.5': 86, '17,0.5': 86, '18,0.5': 86, '19,0.5': 86, '20,0.5': 86, '21,0.5': 86, '22,0.5': 86, '23,0.5': 86, '24,0.5': 85, '24.5,1': 85, '24,1.5': 85, '23.5,1': 85, '12.5,2': 87, '13,1.5': 87, '13.5,1': 87}

    count = 0
    for r0, c0 in l_rc:
        if map_done.get(r0, c0) is not None:
            continue

        count += 1
        if count > max_count:
            break

        # print(r0, c0)

        index = len(conts)

        cont = [[c0, r0]]
        l_sub_blob, l_prefer_sub_blob, l_branch = [], [], []

        for de in [0, 1]:
            cont, sub_blob, prefer_sub_blob = _trace(cont[::-1], de, index)
            l_sub_blob.append(sub_blob)
            l_prefer_sub_blob.append(prefer_sub_blob)

        # print(cont)

        li_cont_new = [index]

        for de in [0, 1]:
            cont = cont[::-1]
            branch = 'NOT_BRANCH'
            i_cont = map_done.get(*cont[-1][::-1])
            if not l_prefer_sub_blob[de]:
                branch, cont = _switch_branch(cont, de, index)
            l_branch.append(branch)
            if branch == 'BRANCH_OVERRIDE' or branch == 'EXTENDED':
                li_cont_new.append(i_cont)

        # print(l_branch)

        # ブロブがある場合、メインのインデックスはindexとは異なっている
        # ブロブとメイン輪郭が重なる部分は上書きされる
        for sub_blob in l_sub_blob:
            _append_blob(sub_blob)

        # 端のインデックスを上書きしないように
        _is_stepping_on = lambda i_cont: i_cont is not None and i_cont not in li_cont_new
        i_begin, i_end = 0, len(cont)
        if l_prefer_sub_blob[0] or _is_stepping_on(map_done.get(*cont[0][::-1])):
            i_begin += 1
        if l_prefer_sub_blob[1] or _is_stepping_on(map_done.get(*cont[-1][::-1])):
            i_end -= 1

        for c, r in cont[i_begin: i_end]:
            map_done.append(r, c, len(conts))

        # print(map_done.get(17, 12.5))

        conts.append(cont)

    if False:
        import pyperclip
        import textwrap
        text = '''\
            conts = {}
            map_done.dict = {}
        '''.format(conts, map_done.dict)
        text = textwrap.dedent(text)
        pyperclip.copy(text)
        print(text)

    # for r0, c0, cut_i in map_cut_check.get_list():
    #     _switch_check_pattern(r0, c0, cut_i)

    min_len = 5

    conts = [cont for cont in conts if len(cont) >= min_len]
    # conts = [cont for cont in conts if _cont_noise_score(cont) < 0.75]

    # print(conts[13])

    _conts = [np.array(cont) - 0.5 for cont in conts if len(cont) >= min_len]
    conts_tree_view.append_cont(_conts, 'conn')
    return

    li_main_cont = []
    for i_cont, cont in enumerate(conts):
        if len(cont) < min_len:
            continue
        li_main_cont.append(i_cont)

    points = []
    for i_cont, cont in enumerate(conts):
        if len(cont) < min_len:
            continue

        for d in [1, -1]:
            c0, r0 = cont[0 if d == 1 else -1]
            i_cont_branch = map_done.get(r0, c0)
            if (i_cont_branch is None or
                i_cont_branch == i_cont or
                    i_cont_branch not in li_main_cont):
                continue

            l_grad = []
            for j in range(1, 10) if d == 1 else range(-1, -10, -1):
                c1, r1 = cont[j - 1]
                c2, r2 = cont[j]
                r, c = cut_quantize(r1, c1, r2, c2)
                grad = map_grad.get(r - 0.5, c - 0.5)
                if not grad:
                    grad = 0
                l_grad.append(grad)

            grad = np.mean(l_grad)
            if grad > 1:
                continue

            print(c0, r0, grad, l_grad)

            points.append([r0, c0])

    conts_tree_view.append_plot(PlotterPoints(points))


def _cont_noise_score(cont):
    def _mark(r0, c0):
        r1, r2 = max(r0 - 1, 0), min(r0 + 2, rr - 1)
        c1, c2 = max(c0 - 1, 0), min(c0 + 2, cc - 1)
        im_t = img_t[r1: r2, c1: c2]
        im_dt = angle_between_tans(im_t, img_t[r0, c0])

        # (dt, s) = (.3π, 0), (0, 1)
        im_s = -im_dt / (.3 * np.pi) + 1
        im_s = np.where(im_s > 0, im_s, 0)

        classes = []
        for r, c in itertools.product(range(r1, r2), range(c1, c2)):
            if map_cut_check.get(r, c):
                continue
            cls = im_s[r - r1, c - c1] > 0
            classes.append((r, c, cls))
        return classes

    map_class = ImageMap()
    for i in range(len(cont) - 1):
        (c1, r1), (c2, r2) = cont[i], cont[i + 1]
        r0, c0 = cut_quantize(r1, c1, r2, c2)
        r0, c0 = int(r0 - 0.5), int(c0 - 0.5)
        if map_cut_check.get(r0, c0):
            continue
        classes = _mark(r0, c0)
        for r, c, cls in classes:
            _classes = map_class.get(r, c)
            if _classes is None:
                _classes = []
            _classes.append((cls, i, r0, c0))
            map_class.append(r, c, _classes)

    indices = set()
    for r, c, classes in map_class.get_list():
        lj_split = [0]
        for j in range(len(classes) - 1):
            if classes[j + 1][1] - classes[j][1] > 1:
                lj_split.append(j + 1)
        lj_split.append(len(classes))

        l_classes = []
        for j1, j2 in zip(lj_split[:-1], lj_split[1:]):
            l_classes.append(classes[j1: j2])

        for _classes in l_classes:
            l_true, l_false = [], []
            for o in _classes:
                if o[0]:
                    l_true.append(o)
                else:
                    l_false.append(o)
            if l_true and l_false:
                # print(r, c, l_true, l_false)
                for _, i, _, _ in _classes:
                    indices.add(i)
    # print(indices)
    return len(indices) / len(cont)


def plot_grad_edge3():
    img_cut_check = np.zeros_like(img_t)
    for r, c, _ in map_cut_check.get_list():
        img_cut_check[r, c] = 1

    def _mark(r0, c0):
        if img_cut_check[r0, c0]:
            return

        p0 = img_p[r0, c0]
        if p0 < 5:
            return

        r1, r2 = max(r0 - 1, 0), min(r0 + 2, rr - 1)
        c1, c2 = max(c0 - 1, 0), min(c0 + 2, cc - 1)
        im_t = img_t[r1: r2, c1: c2]
        im_dt = angle_between_tans(im_t, img_t[r0, c0])
        im_p = img_p[r1: r2, c1: c2]
        im_dp = im_p / p0

        # (dt, s) = (.3π, 0), (0, 1)
        im_s = -im_dt / (.3 * np.pi) + 1
        im_s = np.where(im_s > 0, im_s, 0)
        im_s = im_s * np.where(im_dp < 0.1, 0, 1)

        im_check = img_cut_check[r1: r2, c1: c2]
        im_s = im_s * (1 - im_check)

        l = []
        i0, _ = quantize_grad_t(img_t[r0, c0])[0]
        for i in [i0, (i0 + 4) % 8]:
            ro, co = get_8d()[i]
            r, c = r0 + ro, c0 + co

            if not box_in(r, c, rr - 1, cc - 1):
                continue

            if im_s[r - r1, c - c1] == 0:
                l.append(1)
                continue

            ratio = img_p[r, c] / img_p[r0, c0]
            if ratio > 1:
                continue
            l.append(1)

        # is peak?
        if len(l) != 2:
            return

        l_edge = {}
        t0 = img_t[r0, c0]
        for de in [1, 0]:
            i = i0 + (2 if de == 0 else -2)
            l_rc, ls, lp = [], [], []
            for di in [-1, 0, 1]:
                r, c = get_8d(r0, c0)[(i + di) % 8]
                if not box_in(r, c, rr - 1, cc - 1):
                    continue
                # if im_s[r - r1, c - c1] == 0:
                #     continue

                ratio = calc_ratio(p0, img_p[r, c])
                if ratio < 1 / 3:
                    continue

                t =  img_t[r, c]

                ro, co = r - r0, c - c0
                dx, dy = co, -ro

                # 法線ベクトル
                tt1 = np.arctan2(-dx, dy)
                tt2 = np.arctan2(dx, -dy)

                dt1 = angle_between_tans(tt1, [t0, t])
                dt2 = angle_between_tans(tt2, [t0, t])

                # 法線ベクトルからのずれ具合
                dt = max(dt1 if min(list(dt1) + list(dt2)) in dt1 else dt2)
                score = dt / (np.pi / 2)

                # 0に近いほど良い
                if score > 0.6:
                    continue

                l_rc.append([r, c])
                ls.append(score)
                lp.append(img_p[r, c])
            if not l_rc:
                continue
            lp = np.array(lp) / np.mean(lp)
            ls = 1 - np.array(ls) + lp
            r, c = l_rc[np.argmax(ls)]
            l_edge[de] = [r, c]

        # print(im_s)
        l_grad = []
        for r, c in get_rc(r1, r2, c1, c2):
            if r == r0 and c == c0:
                continue
            if im_s[r - r1, c - c1] == 0:
                continue
            l_grad.append([r, c])

        return l_edge, l_grad

    l_rc = get_rc(0, rr - 1, 0, cc - 1)

    map_edges = ImageMap()
    map_grads = ImageMap()
    for r0, c0 in l_rc:
        ret = _mark(r0, c0)
        if not ret:
            continue

        edges, grads = ret
        map_edges.append(r0, c0, edges)
        map_grads.append(r0, c0, grads)

    conts = []
    for r0, c0, edge in map_edges.get_list():
        for de in [1, 0]:
            if de not in edge:
                continue
            r, c = edge[de]
            conts.append([[c0, r0], [(c0 + c) / 2, (r0 + r) / 2]])

    conts_tree_view.append_cont(conts, 'peak')

    # l_rc = get_rc(1, rr - 1, 1, cc - 1)
    # # l_rc = [[7, 20]]

    # def _is_blob(line):
    #     l_li = connect_line_bb_circular(line)
    #     if not l_li:
    #         return False
    #     li = max(l_li, key=lambda li: len(li))
    #     # print(li)
    #     return len(li) >= 5

    # points = []
    # for r0, c0 in l_rc:
    #     v0 = img_b[r0, c0]
    #     lv = [img_b[r, c] for r, c in get_8d(r0, c0)]
    #     if _is_blob(np.where(lv < v0 - 4, 1, 0)):
    #         points.append([r0, c0])
    #     elif _is_blob(np.where(lv > v0 + 4, 1, 0)):
    #         points.append([r0, c0])

    # conts_tree_view.append_plot(PlotterPoints(points))

    def _check_curvature(cont):
        if len(cont) < 3:
            return True
        if abs(interior_angle(*cont[-3:])) > np.pi / 2:
            cont.pop(-1)
            return False
        return True

    def _trace(cont, de):
        c0, r0 = cont[-1]
        if map_i_cont.get(r0, c0) is not None:
            return False

        edges, grads = map_edges.get(r0, c0), map_grads.get(r0, c0)
        if not edges or not grads:
            return False

        if de not in edges:
            return False

        if not _check_curvature(cont):
            return False

        r1, c1 = edges[de]
        # ピークあり
        if map_edges.get(r1, c1):
            cont.append([c1, r1])
            return True

        # ピークなし
        # スライドできるピークはあるか？
        i0 = get_8d(r0, c0).index([r1, c1])
        for i in [i0 - 1, i0 + 1]:
            i = i % 8
            r1, c1 = get_8d(r0, c0)[i]
            if [r1, c1] not in grads:
                continue
            if not map_edges.get(r1, c1):
                continue
            cont.append([c1, r1])
            return True

        return False

    def _switch_branch(cont0, de):
        def _cont_contrast(cont):
            return [img_p[r, c] for c, r in cont]
        def _score_curvature(cont):
            l_ti = []
            for j1, j2 in range_adj(len(cont)):
                c1, r1 = cont[j1]
                c2, r2 = cont[j2]
                i = get_8d().index([r2 - r1, c2 - c1])
                l_ti.append(i)

            dt = chaincode_dif_abs(l_ti[0], l_ti[-1])
            # (dt, score) = (0, 1), (1, 1), (2, 0.5), (3, 0), (4, 0)
            return max(min((3 - dt) / 2, 1), 0)

        def _score(cont):
            lv = _cont_contrast(cont[1:])
            # print(lv)
            return np.sum(lv)

        c, r = cont0[-1]
        i_cont = map_i_cont.get(r, c)
        if i_cont is None:
            return 'NOT_BRANCH', cont0

        cont1 = conts[i_cont]

        try:
            i = cont1.index([c, r])
        except Exception as e:
            print(e)
            return 'NOT_BRANCH', cont0

        # print(de, cont0)
        # print(i, cont1)
        # print()

        if de == 0 and i == len(cont1) - 1:
            contrast1 = np.mean(_cont_contrast(cont0[-5:-1]))
            contrast2 = np.mean(_cont_contrast(cont1[-1:-5:-1]))
            if calc_ratio(contrast1, contrast2) > 0.5:
                cont = cont0 + cont1[::-1][1:]
                del cont1[:]
                return 'EXTENDED', cont
            else:
                return 'NOT_BRANCH', cont0
        if de == 1 and i == 0:
            contrast1 = np.mean(_cont_contrast(cont0[-5:-1]))
            contrast2 = np.mean(_cont_contrast(cont1[:4]))
            if calc_ratio(contrast1, contrast2) > 0.5:
                cont = cont0 + cont1[1:]
                del cont1[:]
                return 'EXTENDED', cont
            else:
                return 'NOT_BRANCH', cont0

        sample_cont0 = cont0[::-1][:4]

        if len(sample_cont0) < 4:
            return 'BRANCH_BUT_NO_CHANGE', cont0

        if de == 0:
            sample_cont1 = cont1[i:][:4]
        else:
            sample_cont1 = cont1[:i + 1][::-1][:4]

        sample_conts = [sample_cont0, sample_cont1]

        ls1 = np.array([_score(cont) for cont in sample_conts])
        max_score = max(ls1)
        if max_score > 0:
            ls1 = ls1 / max_score

        ls2 = np.array([_score_curvature(cont) for cont in sample_conts])

        score0, score1 = ls1 + ls2 * 0.7

        # print(sample_cont0, sample_cont1)
        # print(ls1, ls2)
        # print(score0, score1)
        # print()

        if score0 > score1:
            if de == 0:
                cont = cont0 + cont1[:i][::-1]
                del cont1[:i]
                return 'BRANCH_OVERRIDE', cont
            else:
                cont = cont0 + cont1[i + 1:]
                del cont1[i + 1:]
                return 'BRANCH_OVERRIDE', cont
        return 'BRANCH_BUT_NO_CHANGE', cont0

    def _2switch_branch(cont0, de):
        c, r = cont0[-1]
        i_cont = map_i_cont.get(r, c)
        if i_cont is None:
            return 'NOT_BRANCH', cont0

        cont1 = conts[i_cont]

        try:
            i = cont1.index([c, r])
        except Exception as e:
            print(e)
            return 'NOT_BRANCH', cont0

        # print(de, cont0)
        # print(i, cont1)
        # print()

        if de == 0 and i == len(cont1) - 1:
            cont = cont0 + cont1[::-1][1:]
            del cont1[:]
            return 'EXTENDED', cont
        if de == 1 and i == 0:
            cont = cont0 + cont1[1:]
            del cont1[:]
            return 'EXTENDED', cont

        if de == 0:
            sample_cont1 = cont1[i:]
        else:
            sample_cont1 = cont1[:i + 1][::-1]

        sample_conts = [cont0, sample_cont1]

        score0, score1 = [len(cont) for cont in sample_conts]

        # print(score0, score1)
        # print()

        if score0 > score1:
            if de == 0:
                cont = cont0 + cont1[:i][::-1]
                del cont1[:i]
                return 'BRANCH_OVERRIDE', cont
            else:
                cont = cont0 + cont1[i + 1:]
                del cont1[i + 1:]
                return 'BRANCH_OVERRIDE', cont
        return 'BRANCH_BUT_NO_CHANGE', cont0

    def _remove_duplicate_point(cont):
        for i in range(len(cont) - 2, -1, -1):
            if cont[i] == cont[i + 1]:
                del cont[i]

    def _dilate_cont(cont, d, i_start, i_end):
        # dj > 2: 鋭角, dj < -2: 鈍角
        def _dj(j1, j2):
            dj = chaincode_dif_clock(j1, j2)
            dj = dj - 8 if dj > 4 else dj
            return -d * dj

        if i_start <= i_end:
            i_start = max([i_start, 0])
            i_end = min([i_end, len(cont)])
            li = range(i_start, i_end - 1)
            i_d = 1
        else:
            i_start = min([i_start, len(cont) - 1])
            i_end = max([i_end, -1])
            li = range(i_start, i_end + 1, -1)
            i_d = -1

        cont_dilate = []

        l_8d = get_8d()
        for i in li:
            (c1, r1), (c2, r2) = cont[i], cont[i + i_d]
            j = l_8d.index([r2 - r1, c2 - c1])

            dj1, dj2 = 0, 0

            im1 = i - i_d
            if im1 >= 0 and im1 < len(cont):
                cm1, rm1 = cont[im1]
                jm1 = l_8d.index([r1 - rm1, c1 - cm1])
                dj1 = _dj(jm1, j)

            i3 = i + 2 * i_d
            if i3 >= 0 and i3 < len(cont):
                c3, r3 = cont[i3]
                j2 = l_8d.index([r3 - r2, c3 - c2])
                dj2 = _dj(j, j2)

            #print(r1, c1, dj1, dj2)

            if dj1 > 2 or dj2 > 2:
                continue

            if j % 2 == 0:
                ro, co = l_8d[(j - d * 2) % 8]
                ps = [[c1 + co, r1 + ro], [c2 + co, r2 + ro]]
                if (i_d > 0 and i == 0) or (i_d < 0 and i == len(cont) - 1) or dj1 >= 2:
                    del ps[0]
                if (i_d > 0 and i == len(cont) - 2) or (i_d < 0 and i == 1) or dj2 >= 2:
                    del ps[-1]

                # ps = [p for p in ps if p not in cont]
                if len(ps) == 0:
                    continue

                # print('a', r1, c1)
                cont_dilate.extend(ps)
            else:
                ro, co = l_8d[(j - d) % 8]
                p = [c1 + co, r1 + ro]
                if p in cont:
                    continue

                # print('s', r1, c1)
                cont_dilate.append(p)

                # 鈍角の場合は回り込むために追加点が必要
                if dj2 <= -2:
                    cont_dilate.append([c2 + co, r2 + ro])

        _remove_duplicate_point(cont_dilate)

        return cont_dilate

    def _dilate_clustering(cont, cont_dilate, map_class, li_rc):
        def _i0_approx(i1):
            return i1 + int((len(cont) - len(cont_dilate)) / 2)

        for i1, p1 in enumerate(cont_dilate):
            c1, r1 = p1
            if not box_in(r1, c1, rr - 1, cc - 1):
                continue

            i0 = _i0_approx(i1)
            ld = []
            for i0 in [i0 - 1, i0, i0 + 1]:
                if i0 < 0 or i0 >= len(cont):
                    continue
                d = distance(cont[i0], p1)
                ld.append([i0, d])
            i0, _ = min(ld, key=lambda o: o[1])

            c0, r0 = cont[i0]

            dt = angle_between_tans(img_t[r0, c0], img_t[r1, c1])

            # (dt, s) = (.3π, 0), (0, 1)
            s = -dt / (.3 * np.pi) + 1
            cls = s > 0 and not map_cut_check.get(r1, c1)
            cls = cls and img_p[r1, c1] / img_p[r0, c0] > 0.5

            map_class.append(r1, c1, cls)

            if not cls:
                if i0 not in li_rc:
                    li_rc[i0] = []
                li_rc[i0].append([r1, c1])


    class TraceResult:
        def __init__(self):
            self.cont = []
            self.map_class = ImageMap()
            self.score = 0

    def _main_trace(result: TraceResult, de):
        cont = result.cont
        map_class = result.map_class

        for _ in range(5):
            if not _trace(cont, de):
                break

        li_rc = dict()
        for d in [1, -1]:
            li_rc[d] = dict()
            cont_dilate0 = cont
            for _ in range(2):
                cont_dilate = _dilate_cont(cont_dilate0, d, 0, len(cont_dilate0))
                _dilate_clustering(cont, cont_dilate, map_class, li_rc[d])
                print(cont_dilate0, cont_dilate, li_rc[d])
                cont_dilate0 = cont_dilate

        length = np.infty
        for d in li_rc:
            length = min(length, len(li_rc[d]))
        result.score = min(1, (length + 2) / len(cont))

    def _main(r0, c0):
        cont = [[c0, r0]]
        for de in [1, 0]:
            for _ in range(100):
                if not _trace(cont, de):
                    break
            cont = cont[::-1]

        # for de in [0, 1]:
        #     _, cont = _switch_branch(cont[::-1], de)

        map_class = ImageMap()

        for d in [1, -1]:
            cont_dilate0 = cont
            for _ in range(2):
                cont_dilate = _dilate_cont(cont_dilate0, d, 0, len(cont_dilate0))
                print(cont_dilate)
                _dilate_clustering(cont, cont_dilate, map_class)
                cont_dilate0 = cont_dilate

        max_step = 500
        l_rc = []

        map_done = ImageMap()

        return cont, map_class

        li_rc_right, li_rc_left = dict(), dict()
        def _add_ne_li(i_cont, r, c):
            if i_cont != len(cont) - 1:
                c0, r0 = cont[i_cont]
                c1, r1 = cont[i_cont + 1]
                t_cont = calc_angle(r0, c0, r1, c1)
            elif i_cont != 0:
                c0, r0 = cont[i_cont]
                c1, r1 = cont[i_cont - 1]
                t_cont = calc_angle(r1, c1, r0, c0)
            else:
                raise Exception()

            t = calc_angle(r0, c0, r, c)
            rot = angle_of_rotation(t_cont, t)
            if rot > 0:
                if i_cont not in li_rc_left:
                    li_rc_left[i_cont] = []
                if [r, c] not in li_rc_left[i_cont]:
                    li_rc_left[i_cont].append([r, c])
            elif rot < 0:
                if i_cont not in li_rc_right:
                    li_rc_right[i_cont] = []
                if [r, c] not in li_rc_right[i_cont]:
                    li_rc_right[i_cont].append([r, c])

        for c0, r0 in cont:
            map_class.append(r0, c0, True)
            map_done.append(r0, c0, True)

        if len(cont) < 2:
            return [], map_class

        for i, (c0, r0) in enumerate(cont):
            grads = map_grads.get(r0, c0)
            for r1, c1 in get_8d(r0, c0):
                if not box_in(r1, c1, rr - 1, cc - 1):
                    continue
                if [r1, c1] not in grads:
                    map_class.append(r1, c1, False)
                    _add_ne_li(i, r1, c1)
                    continue
                if map_done.get(r1, c1):
                    continue
                l_rc.append([r1, c1])

        for step in range(max_step):
            if not l_rc:
                break
            r0, c0 = l_rc.pop(0)

            if map_done.get(r0, c0):
                continue
            map_done.append(r0, c0, True)

            l_sli, _, l_d = suture_line_i(cont, [[c0, r0]], th_k = 5)
            if len(l_d) == 0 or l_d[0] > 2:
                continue

            i_cont = l_sli[0][0]
            c_base, r_base = cont[i_cont]
            dt = angle_between_tans(img_t[r_base, c_base], img_t[r0, c0])

            # (dt, s) = (.3π, 0), (0, 1)
            s = -dt / (.3 * np.pi) + 1
            cls = s > 0 and not map_cut_check.get(r0, c0)
            cls = cls and img_p[r0, c0] / img_p[r_base, c_base] > 0.5

            map_class.append(r0, c0, cls)

            if not cls:
                _add_ne_li(i_cont, r0, c0)
                continue

            for r1, c1 in get_8d(r0, c0):
                if not box_in(r1, c1, rr - 1, cc - 1):
                    continue
                if map_done.get(r1, c1):
                    continue
                l_rc.append([r1, c1])

        if step == max_step - 1:
            print('finish: step =', step)

        def _li_interpolate(li_rc):
            li = sorted(li_rc.keys())

            for j in range(len(li) - 1):
                if li[j + 1] - li[j] == 1:
                    continue
                l_rc1 = li_rc[li[j]]
                l_rc2 = li_rc[li[j + 1]]
                found = False
                for (r1, c1), (r2, c2) in itertools.product(l_rc1, l_rc2):
                    if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                        found = True
                        break
                if found:
                    for k in range(li[j] + 1, li[j + 1]):
                        li_rc[k] = []

            return sorted(li_rc.keys())

        def _l_li(li):
            return list(filter(lambda li: len(li) >= 3, connect_line(list(li), tolerance=2)))
        def _max_li(l_li):
            return max(l_li, key=lambda li: li[-1] - li[0])

        li_right = _li_interpolate(li_rc_right)
        li_left = _li_interpolate(li_rc_left)

        l_li_right = _l_li(li_right)
        l_li_left = _l_li(li_left)

        # print(l_li_right, l_li_left)

        valid = False
        if l_li_right and l_li_left:
            li_right = _max_li(l_li_right)
            li_left = _max_li(l_li_left)

            if len(li_right) > len(li_left):
                li = li_right
                l_li_sub = l_li_left
            else:
                li = li_left
                l_li_sub = l_li_right

            # print(li, l_li_sub)

            count = 0
            for li_sub in l_li_sub:
                if li_sub[0] <= li[0] and li[0] <= li_sub[-1]:
                    count += li_sub[-1] - li[0]
                elif li_sub[0] <= li[-1] and li[-1] <= li_sub[-1]:
                    count += li[-1] - li_sub[0]
                elif li[0] < li_sub[0] and li_sub[-1] < li[-1]:
                    count += li_sub[-1] - li_sub[0]

            if count / len(li) > 0.5:
                valid = True
                cont = cont[li[0]: li[-1] + 1]

        if not valid:
            cont = []

        return cont, map_class

    conts = []
    map_i_cont = ImageMap()
    map_done = ImageMap()

    max_count = 500
    count = 0

    l_rc = map_edges.get_list()
    l_rc = [[1, 19, 0]]

    if False:
        max_count = 1
        l_rc = [[12, 17, 0]]
        conts = [[[5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [4, 7], [4, 8]], [[8, 0], [8, 1], [7, 2], [7, 3], [7, 4]], [[2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [1, 15], [1, 16], [1, 17], [1, 18]], [[4, 5], [4, 4], [4, 3], [4, 2], [4, 1]], [[10, 2], [11, 2], [12, 1], [13, 1]], [[19, 11], [19, 10], [19, 9], [19, 8], [19, 7], [19, 6], [18, 5], [18, 4], [19, 3], [19, 2], [19, 1]], [[20, 1], [20, 2], [20, 3], [20, 4], [20, 5], [20, 6], [20, 7], [20, 8], [20, 9], [20, 10], [20, 11]], [[22, 3], [23, 2], [24, 2]], [[11, 7], [11, 6], [11, 5]], [[13, 5], [12, 5], [11, 5]], [[15, 5], [15, 6], [15, 7]], [[5, 9], [6, 9], [7, 9], [8, 9], [9, 9], [10, 9], [11, 8]], [[23, 7], [24, 8], [24, 9], [24, 10], [23, 11], [23, 12]], [[4, 21], [4, 20], [3, 19], [3, 18], [3, 17], [3, 16], [3, 15], [3, 14], [3, 13], [3, 12], [4, 11]], [[21, 11], [22, 11], [23, 12]]]
        map_i_cont.dict = {'0,5': 0, '1,5': 0, '2,5': 0, '3,5': 0, '4,5': 0, '5,5': 0, '6,5': 0, '7,4': 0, '8,4': 0, '0,8': 1, '1,8': 1, '2,7': 1, '3,7': 1, '4,7': 1, '1,2': 2, '2,2': 2, '3,2': 2, '4,2': 2, '5,2': 2, '6,2': 2, '7,2': 2, '8,2': 2, '9,2': 2, '10,2': 2, '11,2': 2, '12,2': 2, '13,2': 2, '14,2': 2, '15,1': 2, '16,1': 2, '17,1': 2, '18,1': 2, '5,4': 3, '4,4': 3, '3,4': 3, '2,4': 3, '1,4': 3, '2,10': 4, '2,11': 4, '1,12': 4, '1,13': 4, '11,19': 5, '10,19': 5, '9,19': 5, '8,19': 5, '7,19': 5, '6,19': 5, '5,18': 5, '4,18': 5, '3,19': 5, '2,19': 5, '1,19': 5, '1,20': 6, '2,20': 6, '3,20': 6, '4,20': 6, '5,20': 6, '6,20': 6, '7,20': 6, '8,20': 6, '9,20': 6, '10,20': 6, '11,20': 6, '3,22': 7, '2,23': 7, '2,24': 7, '7,11': 8, '6,11': 8, '5,11': 9, '5,13': 9, '5,12': 9, '5,15': 10, '6,15': 10, '7,15': 10, '9,5': 11, '9,6': 11, '9,7': 11, '9,8': 11, '9,9': 11, '9,10': 11, '8,11': 11, '7,23': 12, '8,24': 12, '9,24': 12, '10,24': 12, '11,23': 12, '12,23': 14, '21,4': 13, '20,4': 13, '19,3': 13, '18,3': 13, '17,3': 13, '16,3': 13, '15,3': 13, '14,3': 13, '13,3': 13, '12,3': 13, '11,4': 13, '11,21': 14, '11,22': 14}
        map_done.dict = {'0,5': True, '1,5': True, '2,5': True, '3,5': True, '4,5': True, '5,5': True, '6,5': True, '7,4': True, '7,5': True, '8,4': True, '8,6': True, '0,6': True, '0,7': True, '1,6': True, '2,6': True, '3,6': True, '0,8': True, '0,9': True, '1,8': True, '1,9': True, '2,7': True, '2,8': True, '3,7': True, '4,7': True, '4,8': True, '4,9': True, '5,8': True, '6,7': True, '1,2': True, '2,2': True, '3,2': True, '4,2': True, '5,1': True, '5,2': True, '6,1': True, '6,2': True, '7,2': True, '8,2': True, '9,1': True, '9,2': True, '9,3': True, '9,4': True, '10,2': True, '10,3': True, '11,2': True, '12,2': True, '13,2': True, '14,2': True, '15,1': True, '16,1': True, '17,1': True, '18,1': True, '19,2': True, '0,4': True, '1,4': True, '2,4': True, '3,4': True, '4,3': True, '4,4': True, '5,4': True, '1,10': True, '1,11': True, '1,12': True, '1,13': True, '2,9': True, '2,10': True, '2,11': True, '2,12': True, '0,15': True, '0,16': True, '0,17': True, '0,18': True, '1,14': True, '1,15': True, '1,16': True, '1,17': True, '1,18': True, '2,16': True, '2,17': True, '2,18': True, '3,17': True, '0,19': True, '1,19': True, '2,19': True, '3,18': True, '3,19': True, '4,18': True, '4,19': True, '5,18': True, '5,19': True, '6,18': True, '6,19': True, '7,18': True, '7,19': True, '8,17': True, '8,18': True, '8,19': True, '9,17': True, '9,18': True, '9,19': True, '10,17': True, '10,18': True, '10,19': True, '11,17': True, '11,18': True, '11,19': True, '0,20': True, '1,20': True, '2,20': True, '3,20': True, '4,20': True, '5,20': True, '6,20': True, '7,20': True, '8,20': True, '9,20': True, '10,20': True, '11,20': True, '2,13': True, '0,25': True, '1,24': True, '1,25': True, '1,26': True, '2,22': True, '2,23': True, '2,24': True, '2,25': True, '2,26': True, '3,22': True, '3,23': True, '3,24': True, '3,25': True, '4,22': True, '4,23': True, '4,24': True, '5,22': True, '3,8': True, '3,9': True, '3,10': True, '4,10': True, '4,11': True, '5,7': True, '5,9': True, '5,10': True, '6,6': True, '6,8': True, '6,9': True, '7,6': True, '7,7': True, '8,7': True, '9,6': True, '9,7': True, '10,7': True, '3,11': True, '3,12': True, '5,12': True, '3,13': True, '4,13': True, '4,14': True, '5,14': True, '6,13': True, '3,21': True, '4,12': True, '3,26': True, '4,25': True, '4,26': True, '5,25': True, '5,26': True, '6,23': True, '6,24': True, '6,25': True, '6,26': True, '7,23': True, '7,24': True, '7,25': True, '8,23': True, '8,24': True, '8,25': True, '8,26': True, '9,26': True, '5,11': True, '6,11': True, '6,12': True, '7,11': True, '7,12': True, '5,13': True, '3,15': True, '4,16': True, '5,15': True, '6,15': True, '6,16': True, '7,13': True, '7,14': True, '7,15': True, '7,16': True, '4,15': True, '5,16': True, '4,21': True, '5,21': True, '5,23': True, '5,24': True, '6,21': True, '6,22': True, '7,21': True, '7,22': True, '6,3': True, '7,3': True, '7,9': True, '8,3': True, '8,9': True, '8,10': True, '8,11': True, '8,12': True, '8,13': True, '8,16': True, '9,10': True, '9,11': True, '9,12': True, '9,13': True, '9,14': True, '9,15': True, '9,16': True, '10,11': True, '10,12': True, '10,13': True, '10,14': True, '10,15': True, '10,16': True, '11,12': True, '11,13': True, '11,14': True, '11,15': True, '11,16': True, '8,21': True, '8,22': True, '9,21': True, '9,22': True, '10,21': True, '11,21': True, '9,5': True, '9,8': True, '9,9': True, '10,5': True, '10,6': True, '10,8': True, '10,9': True, '10,10': True, '11,5': True, '11,7': True, '11,8': True, '9,23': True, '9,24': True, '10,23': True, '10,24': True, '10,26': True, '11,23': True, '11,24': True, '12,23': True, '12,24': True, '11,1': True, '10,4': True, '11,4': True, '11,6': True, '12,3': True, '12,4': True, '12,5': True, '12,6': True, '13,3': True, '13,4': True, '14,3': True, '14,4': True, '15,2': True, '15,3': True, '16,2': True, '16,3': True, '16,4': True, '17,3': True, '18,3': True, '19,3': True, '19,4': True, '20,3': True, '20,4': True, '20,5': True, '21,4': True, '21,5': True, '21,6': True, '11,22': True}

    results: 'list[TraceResult]' = []
    while l_rc:
        r0, c0, _ = l_rc.pop(0)
        if count >= max_count:
            break

        if map_done.get(r0, c0):
            continue

        result = TraceResult()
        result.cont.append([c0, r0])

        for de in [0, 1]:
            _main_trace(result, de)
            result.cont = result.cont[::-1]

        cont = result.cont
        map_class = result.map_class

        for c, r in cont:
            map_i_cont.append(r, c, len(conts))

        #if len(cont) > 3 and result.score > 0.5:
        if cont:
            print('i_cont = {:2}, r0 = {}, c0 = {}, score = {}'.format(len(conts), r0, c0, result.score))
            conts.append(cont)
            results.append(result)
            count += 1

        for c, r in cont:
            map_done.append(r, c, True)

        for r, c, cls in map_class.get_list():
            if cls:
                map_done.append(r, c, True)

    if False:
        import pyperclip
        import textwrap
        text = '''\
            conts = {}
            map_i_cont.dict = {}
            map_done.dict = {}
        '''.format(conts, map_i_cont.dict, map_done.dict)
        text = textwrap.dedent(text)
        pyperclip.copy(text)
        print(text)

    def _print_map_i_cont():
        print('  ', ' '.join(['{:2}'.format(v) for v in range(cc - 1)]))
        for r in range(rr - 1):
            l = []
            for c in range(cc - 1):
                v = map_i_cont.get(r, c)
                v = v if v else 0
                l.append('{:2}'.format(v))
            print('{:2}'.format(r), ' '.join(l))

    _print_map_i_cont()

    cont_relations = []

    for i_cont0, result in enumerate(results):
        set_i_cont = set()
        for r, c, cls in result.map_class.get_list():
            if cls:
                continue
            i_cont = map_i_cont.get(r, c)
            if i_cont is not None:
                set_i_cont.add(i_cont)
        for i_cont in set_i_cont:
            pair = (i_cont0, i_cont) if i_cont0 < i_cont else (i_cont, i_cont0)
            if pair not in cont_relations:
                cont_relations.append(pair)

    cont_relations = sorted(cont_relations)

    for pair in cont_relations:
        l_sli, _, l_d = suture_line_i(*[conts[i] for i in pair])
        if len(l_d) > 3:
            print(pair, l_d)


    conts_tree_view.append_cont(conts, 'peak-conns')

    plotters = []
    for i, result in enumerate(results):
        xs, ys, cs = [], [], []
        for r, c, cls in result.map_class.get_list():
            xs.append(c + 0.5)
            ys.append(r + 0.5)
            cs.append('red' if cls else 'blue')
        plotters.append(PlotterScatter(xs, ys, cs, 5))
    conts_tree_view.append_plots(plotters, 'class')


img_edge_mask = np.zeros((rr, cc), dtype=np.uint8)
map_cut_2x2 = ImageMap['list[Cut2x2]']()
map_cut_check = ImageMap()

if not img_only:
    plot_edge_2x2()
    # plot_non_maximum_suppression()
    # plot_grad_edge()
    # plot_grad_edge2()
    plot_grad_edge3()
    # plot_corner()
    # plot_edge_peak()
    # plot_peak_line()
    # plot_edge()
    # plot_grad_area()
    # plot_depression_palm()
    # plot_hoge()
    pass

# ax[0].imshow(img)
ax[0].imshow(img_b, cmap='gray', vmin=0, vmax=0xff)
ax[1].imshow(img_b, cmap='gray', vmin=0, vmax=0xff)


def get_axis_lim(ax):
    return [ax.get_xlim(), ax.get_ylim()]


def on_press(event):
    global last_lim
    last_lim = get_axis_lim(ax[0])


def on_release(event):
    l_lim = [get_axis_lim(ax[i]) for i in [0, 1]]

    i = int(last_lim == l_lim[1])
    xlim, ylim = l_lim[1 - i]

    ax[i].set_xlim(xlim)
    ax[i].set_ylim(ylim)

    if l_lim[0] != l_lim[1]:  # changed
        c1, c2, r2, r1 = [int(v + 0.5) for v in xlim + ylim]
        print('[{}: {}, {}: {}]'.format(r1, r2 + 1, c1, c2 + 1))


last_lim = get_axis_lim(ax[0])

conts_tree_view.canvas.figure.canvas.mpl_connect(
    'button_press_event', on_press)
conts_tree_view.canvas.figure.canvas.mpl_connect(
    'button_release_event', on_release)

if not img_only:
    for a in ax:
        a.set_xticks(range(cc - 1, -1, -2))
        a.set_yticks(range(rr - 1, -1, -2))
        a.set_xlim(-0.5, cc - 0.5)
        a.set_ylim(rr - 0.5, -0.5)

print('', flush=True)

conts_tree_view.canvas.figure.tight_layout()
conts_tree_view.show()
plt.show()
sys.exit(app.exec_())

"""
"""
