import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_csv(filename: str) -> list[tuple]:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        return [(int(row[0]), int(row[1])) for row in reader]


def update(i, x, y):
    line.set_data(x[:i], y[:i])
    return line,

def check(l1, l2):
    for i in range(len(gpu_points)):
        x1, y1 = l1[i]
        x2, y2 = l2[i]
        if x1!=x2 or y1!=y2:
            print(f'not equal index {i}')

if __name__ == "__main__":
    gpu_points = read_csv("cities-csv/sorted_cities_gpu.csv")
    cpu_points = read_csv("cities-csv/sorted_cities_cpu.csv")
    check(gpu_points, cpu_points)
    fig, ax = plt.subplots()
    line, = ax.plot(*zip(*gpu_points))
    x, y = zip(*gpu_points)
    ax.scatter(x, y)
    anim = FuncAnimation(fig, update, frames=len(gpu_points)+1, fargs=(x, y), interval=100)
    plt.show()