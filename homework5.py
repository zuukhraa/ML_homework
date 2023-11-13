import random
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np


class Point:
    def __init__(self, x, y, class_index):
        self.x = x
        self.y = y
        self.class_index = class_index


def generate_points(num_points):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        class_index = random.randint(0, 1)
        points.append(Point(x, y, class_index))
    return points

def plot_points(points):
    class_0 = [point for point in points if point.class_index == 0]
    class_1 = [point for point in points if point.class_index == 1]

    plt.scatter([point.x for point in class_0], [point.y for point in class_0], color='red', label='Class 0')
    plt.scatter([point.x for point in class_1], [point.y for point in class_1], color='blue', label='Class 1')

    plt.legend()
    plt.grid(True)
    plt.show()

points = generate_points(100)
plot_points(points)

def plot_points_with_line(points, clf):
    class_0 = [point for point in points if point.class_index == 0]
    class_1 = [point for point in points if point.class_index == 1]

    plt.scatter([point.x for point in class_0], [point.y for point in class_0], color='red', label='Class 0')
    plt.scatter([point.x for point in class_1], [point.y for point in class_1], color='blue', label='Class 1')

    plt.legend()

    # Определение коэффициентов прямой
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 10)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    plt.plot(xx, yy, 'k-', label='Decision Boundary')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.show()

# Обучение модели и построение графика с прямой
clf = svm.SVC(kernel='linear', C=1000)
X = [[point.x, point.y] for point in points]
y = [point.class_index for point in points]
clf.fit(X, y)

plot_points_with_line(points, clf)

def plot_points_with_new_point(points, new_point):
    class_0 = [point for point in points if point.class_index == 0]
    class_1 = [point for point in points if point.class_index == 1]

    plt.scatter([point.x for point in class_0], [point.y for point in class_0], color='red', label='Class 0')
    plt.scatter([point.x for point in class_1], [point.y for point in class_1], color='blue', label='Class 1')
    plt.scatter(new_point.x, new_point.y, color='green', label='New Point')

    plt.legend()
    plt.grid(True)
    plt.show()

# Создание новой точки с случайными координатами, класс не определен
new_point = Point(random.uniform(0, 10), random.uniform(0, 10), None)

# Добавление новой точки в массив
points.append(new_point)

# Отображение точек на графике, включая новую точку
plot_points_with_new_point(points, new_point)

# Определение класса новой точки с помощью обученного классификатора
new_point.class_index = clf.predict([[new_point.x, new_point.y]])[0]

# Отображение точек на графике с новым классом новой точки
plot_points_with_line(points, clf)