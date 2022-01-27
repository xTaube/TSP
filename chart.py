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


if __name__ == "__main__":
    gpu_points = read_csv("sorted_cities_gpu.csv")
    cpu_points = read_csv("sorted_cities_cpu.csv")
    fig, ax = plt.subplots()
    line, = ax.plot(*zip(*gpu_points))
    x, y = zip(*gpu_points)
    ax.scatter(x, y)
    anim = FuncAnimation(fig, update, frames=len(gpu_points)+1, fargs=(x, y), interval=1000)
    plt.show()