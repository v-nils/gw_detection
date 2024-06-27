import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


def get_path(id: str) -> str:
    """
    Get the path of the Gravitational Wave data.

    :param id:
    :param train:
    :return:
    """

    return f'/train/{id[0]}/{id[1]}/{id[2]}/{id}.npy'


def create_2_6_plot():
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(2, 6, width_ratios=[10, 1, 10, 1, 10, 1], height_ratios=[2, 3], hspace=0.3, wspace=0.2)

    ax_1 = [fig.add_subplot(gs[0, i * 2]) for i in range(3)]
    ax_2 = [fig.add_subplot(gs[0, i * 2 + 1]) for i in range(3)]
    ax_3 = [fig.add_subplot(gs[1, i * 2:(i + 1) * 2]) for i in range(3)]

    return fig, (ax_1, ax_2, ax_3)


def create_1_6_plot():
    fig = plt.figure(figsize=(16, 2.8))
    gs = gridspec.GridSpec(1, 6, width_ratios=[10, 1, 10, 1, 10, 1], hspace=0.3, wspace=0.2)

    ax_1 = [fig.add_subplot(gs[0, i*2:i*2+1]) for i in range(3)]

    return fig, ax_1
