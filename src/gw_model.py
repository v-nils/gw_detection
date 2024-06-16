from types import NoneType
import pywt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from src.utils import get_path
import os
from matplotlib.axes import Axes
import torch
from scipy import signal
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from nnAudio.Spectrogram import CQT1992v2

# -- Global Variables -- #
# -- -- -- -- -- -- -- -- #

plt.style.use(['science', 'ieee'])
gw_root_path: str = r'../data/raw_data/g2net-gravitational-wave-detection'
training_label_path: str = r'../data/raw_data/g2net-gravitational-wave-detection/training_labels.csv'
waveform_plots: str = r'../data/results/plots/waveforms'


class GravWave:
    """
    A class to represent a Gravitational Wave.
    """

    def __init__(self, wave_id: str) -> None:
        """
        Initialize the GW object.

        :param wave_id: ID of the Gravitational Wave.
        """
        self.id: str = wave_id
        self.wave: np.ndarray | None = None
        self.full_wave = None

    def load_data(self) -> None:
        """
        Load the data of the Gravitational Wave.

        :return: None
        """

        # Load the full wave data
        wave_path: str = get_path(self.id, train=True)
        path: str = gw_root_path + wave_path

        print("Get wave from path: ", path)

        self.wave = np.load(path)

    def plot_wave(self,
                  ax_wave: Axes,
                  ax_density: Axes | None = None,
                  ax_image: Axes | None = None,
                  which: int = 0,
                  y_lims: tuple[float] | None = None) -> Axes:
        """
        Plot the wave of the Gravitational Wave.

        :param ax_wave: Axes object to plot the wave of shape.
        :param ax_density: Axes object to plot the density of the wave.
        :param which: Which wave to plot.
        :param y_lims: Y-axis limits.
        :return: Axes object.
        """

        ax_wave.plot(self.wave[which, :])
        ax_wave.set_title(f'Gravitational Wave {self.id}')
        ax_wave.set_xlabel('Time')
        ax_wave.set_ylabel('Amplitude')

        if y_lims:
            ax_wave.set_ylim(y_lims)

        if ax_density:
            sns.kdeplot(self.wave[which, :], ax=ax_density, fill=True, vertical=True)
            ax_density.set_title('Density Plot')
            ax_density.set_xlabel('Density')
            ax_density.set_ylabel('Amplitude')
            ax_density.yaxis.set_label_position('right')

        if ax_image:
            ax_image.imshow(self.full_wave[which, :], aspect='auto', origin='lower')
            ax_image.set_title('CQT Image')
            ax_image.set_xlabel('Time')
            ax_image.set_ylabel('Frequency')

        return ax_wave

    def decompose_wave(self, transform=CQT1992v2(sr=2048, hop_length=64, fmin=20, fmax=500)):
        """
        Decompose the wave into its frequency components.

        :param transform:
        :return:
        """

        g_wave = np.concatenate(self.wave, axis=0)
        bHP, aHP = signal.butter(8, (20, 500), btype='low', fs=2048)
        window = signal.tukey(4096 * 3, 0.2)
        g_wave *= window
        g_wave = signal.filtfilt(bHP, aHP, g_wave)
        g_wave = g_wave / np.max(g_wave)
        g_wave = torch.from_numpy(g_wave).float()
        image = transform(g_wave)
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))

        self.full_wave = image


if __name__ == "__main__":
    wave = GravWave('00001f4945')
    wave.load_data()
    wave.decompose_wave()

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 3, width_ratios=[3, 1, 2])

    for i in range(3):
        ax_wave = fig.add_subplot(gs[i, 0])
        ax_density = fig.add_subplot(gs[i, 1], sharey=ax_wave)
        ax_image = fig.add_subplot(gs[i, 2])

        wave.plot_wave(ax_wave, ax_density=ax_density, ax_image=ax_image, which=i)

    plt.tight_layout()
    plt.savefig(waveform_plots + '/waveform.png')

    wave.full_wave.shape
