import os
import sys
import cv2
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.axes import Axes
import torch
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from nnAudio.Spectrogram import CQT1992v2
from src.utils import get_path, create_1_6_plot, create_2_6_plot
import scienceplots
import librosa
import pywt
from torch.utils.data import DataLoader
from sklearn import model_selection as sk_model_selection
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import torch.utils.data as torch_data
import time
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils import data as torch_data
from torch.nn import functional as torch_functional
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, r2_score

# -- Global Variables -- #
plt.style.use(['science', 'ieee'])
gw_root_path: str = r'../data/raw_data/g2net-gravitational-wave-detection'
training_label_path: str = r'../data/raw_data/g2net-gravitational-wave-detection/training_labels.csv'
waveform_plots: str = r'../data/results/plots/waveforms'

training_labels: pd.DataFrame = pd.read_csv(training_label_path, index_col=0)


@dataclass
class GravWave:
    """
    A class to represent a Gravitational Wave.
    """

    def __init__(self, wave_id: str) -> None:
        """
        Initialize the GW object.

        :param wave_id: ID of the Gravitational Wave.
        """
        self.full_wave_libr = None
        self.id: str = wave_id
        self.wave: np.ndarray | None = None
        self.full_wave_qct = None
        self.full_wave_stft = None
        self.full_wave_mfcc = None
        self.full_wave_wavelet = None
        self.full_wave_cwt = None
        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Load the data of the Gravitational Wave.

        :return: None
        """

        # Load the full wave data
        wave_path: str = get_path(self.id)
        path: str = gw_root_path + wave_path

        self.wave = np.load(path)

    def plot_wave(self,
                  ax_wave: list[Axes],
                  ax_density: list[Axes] | None = None,
                  ax_image: list[Axes] | None = None,
                  transform: str = 'qct',
                  y_lims: tuple[float] | None = None):
        """
        Plot the wave of the Gravitational Wave.

        :param ax_wave: Axes object to plot the wave of shape.
        :param ax_density: Axes object to plot the density of the wave.
        :param ax_image: Axes object to plot the image of the wave.
        :param y_lims: Y-axis limits.
        :return: Axes object.
        """

        if transform == 'qct':
            spectro = [q.squeeze() for q in self.full_wave_qct]
        elif transform == 'stft':
            spectro = [np.abs(s) for s in self.full_wave_stft]
        elif transform == 'mfcc':
            spectro = self.full_wave_mfcc
        elif transform == 'wavelet':
            spectro = self.full_wave_wavelet
        elif transform == 'cwt':
            spectro = self.full_wave_cwt
        else:
            raise ValueError("Invalid transform type.")

        for which in range(3):

            if ax_wave is not None:
                ax_wave[which].plot(self.wave[which, :])
                ax_wave[which].set_title(["Hanford", "Livingston", "Virgo"][which], fontsize=19)
                ax_wave[which].set_xlabel('Time')
                if which == 0:
                    ax_wave[which].set_ylabel('Amplitude', fontsize=12)

                if y_lims:
                    ax_wave[which].set_ylim(y_lims)

            if ax_density is not None:
                sns.kdeplot(y=self.wave[which, :], ax=ax_density[which], fill=True)
                ax_density[which].set_xlabel('Density', fontsize=6)
                ax_density[which].xaxis.set_label_position('bottom')
                ax_density[which].xaxis.set_ticks_position('top')
                ax_density[which].yaxis.set_label_position('right')
                ax_density[which].set_yticklabels([])

            if ax_image is not None:
                ax_image[which].imshow(spectro[which], aspect='auto', origin='lower', cmap='magma_r')
                ax_image[which].set_xlabel('Time')
                if which == 0:
                    ax_image[which].set_ylabel(f'{transform}', fontsize=12)

    def cqt_transform(self):
        if self.full_wave_qct is None:
            self.full_wave_qct = []

        TRANSFORM = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64, verbose=False)
        image = []

        for i in range(3):
            waves = self.wave[i] / np.max(self.wave[i])
            waves = torch.from_numpy(waves).float()

            channel = TRANSFORM(waves).squeeze().numpy()
            image.append(channel)

        # Convert the list of numpy arrays to a single numpy array before converting to a tensor
        image_array = np.array(image)
        self.full_wave_qct = torch.tensor(image_array, dtype=torch.float64)

    def stft_transform(self):
        if self.full_wave_stft is None:
            self.full_wave_stft = []

        for i in range(3):
            stft_result = librosa.stft(self.wave[i, :], n_fft=2048, hop_length=512)
            self.full_wave_stft.append(stft_result)

    def mfcc_transform(self):
        if self.full_wave_mfcc is None:
            self.full_wave_mfcc = []

        for i in range(3):
            mfcc_result = librosa.feature.mfcc(y=self.wave[i, :], sr=2048, n_mfcc=13)
            self.full_wave_mfcc.append(mfcc_result)

    def wavelet_transform(self):
        if self.full_wave_wavelet is None:
            self.full_wave_wavelet = []

        for i in range(3):
            coeffs, _ = pywt.cwt(self.wave[i, :], scales=np.arange(1, 128), wavelet='cmor')
            self.full_wave_wavelet.append(np.abs(coeffs))

    def cwt_transform(self):
        if self.full_wave_cwt is None:
            self.full_wave_cwt = []

        widths = np.arange(1, 128)
        for i in range(3):
            cwt_result, _ = pywt.cwt(self.wave[i, :], scales=widths, wavelet='cmor')
            self.full_wave_cwt.append(np.abs(cwt_result))



def plot_data_balance():
    """
    Plot the data balance of the Gravitational Wave data.

    :return: None
    """
    training_labels: pd.DataFrame = pd.read_csv(training_label_path)
    training_labels['target'] = training_labels['target'].astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    sns.countplot(x='target', data=training_labels, ax=ax)
    ax.set_xlabel('Target', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Data Balance', fontsize=16)
    plt.show()


class DataRetriever(torch_data.Dataset):

    def __init__(self, indexes: list[str], spectrogram: str = 'qct'):
        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")

        self.indexes: list[str] = indexes
        self.spectrogram = spectrogram

        if cuda_available:
            print("CUDA Available. Using", torch.cuda.get_device_name(0))
        else:
            print("CUDA is not available. Using CPU.")

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index: int):

        X_ = GravWave(training_labels.index[index])

        if self.spectrogram == 'qct':
            X_.cqt_transform()
            X = X_.full_wave_qct.to(self.device).type(torch.float32)

        elif self.spectrogram == 'stft':
            X_.stft_transform()
            X = torch.tensor(X_.full_wave_stft, dtype=torch.float32).to(self.device)

        elif self.spectrogram == 'mfcc':
            X_.mfcc_transform()
            X = torch.tensor(X_.full_wave_mfcc, dtype=torch.float32).to(self.device)

        elif self.spectrogram == 'wavelet':
            X_.wavelet_transform()
            X = torch.tensor(X_.full_wave_wavelet, dtype=torch.float32).to(self.device)

        elif self.spectrogram == 'cwt':
            X_.cwt_transform()
            X = torch.tensor(X_.full_wave_cwt, dtype=torch.float32).to(self.device)

        else:
            raise ValueError("Invalid spectrogram type.")

        y = training_labels.target.iloc[index]
        y = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1)  # Ensure the target has shape [batch_size, 1]

        return X, y


class Model(nn.Module):

    def __init__(self, input_shape: tuple[int, int, int] = (3, 69, 65)):
        super(Model, self).__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(64 * (input_shape[1] // 8) * (input_shape[2] // 8), 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, n_bins: int = 69, num_time_steps: int = 65):
        x = self.pool(F. relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * (n_bins // 8) * (num_time_steps // 8))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 epochs: int = 10,
                 batch_size: int = 32,
                 device: str = 'cuda'):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def train(self, train_loader: torch_data.DataLoader, val_loader: torch_data.DataLoader = None):

        self.model.train()
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            running_train_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / len(train_loader)
            train_losses.append(epoch_train_loss)

            if val_loader:
                running_val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        running_val_loss += loss.item()

                epoch_val_loss = running_val_loss / len(val_loader)
                val_losses.append(epoch_val_loss)
                self.model.train()

            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}" if val_loader else f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_train_loss:.4f}")

        # Plot the training and validation loss
        plt.figure()
        plt.plot(range(1, self.epochs + 1), train_losses, label='Training Loss')
        if val_loader:
            plt.plot(range(1, self.epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig('../results/loss/loss.png')
        plt.show()

    def evaluate(self, test_loader: torch_data.DataLoader):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Calculate ROC curve and AUC

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Calculate R^2 score
        r2 = r2_score(y_true, y_pred)

        print(f"R^2 score: {r2}")

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label="ROC curve (area = {:.2f})".format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('R²={:.2f}'.format(r2))
        plt.legend(loc="lower right")
        plt.savefig('../results/roc/roc_curve.png')
        plt.show()

        # Plot the distribution of the predictions
        plt.figure()
        sns.histplot(y_pred, kde=True)
        plt.xlabel('Predictions')
        plt.ylabel('Density')
        plt.title('Distribution of Predictions')
        plt.savefig('../results/distribution/predictions.png')
        plt.show()





if __name__ == "__main__":

    data_exploration: bool = False
    cnn_training: bool = True

    spectro = 'qct'

    if data_exploration:

        plot_data_balance()

        wave = GravWave('fff19c1af3')

        wave.cqt_transform()
        #wave.stft_transform()
        #wave.mfcc_transform()
        #wave.wavelet_transform()
        #wave.cwt_transform()

        wave_transforms = ['qct'] #, 'stft', 'mfcc', 'wavelet', 'cwt']

        for transform in wave_transforms:

            print(f"Plotting the wave with transform: {transform}")

            if transform == 'qct':
                fig, (ax_signal, ax_kde, ax_spectrum) = create_2_6_plot()
            else:
                fig, ax_spectrum = create_1_6_plot()
                ax_signal = None
                ax_kde = None

            wave.plot_wave(ax_wave=ax_signal, ax_density=ax_kde, ax_image=ax_spectrum, transform=transform)
            plt.show()

    if cnn_training:
        idx_1 = training_labels[training_labels.target == 1][:250].index.values
        idx_0 = training_labels[training_labels.target == 0][:250].index.values

        # Combine the indices
        indexes = np.concatenate([idx_1, idx_0])

        # Split indexes into training+validation and test sets
        train_val_indexes, test_indexes = train_test_split(indexes, test_size=0.2, random_state=42)

        # Further split training+validation into training and validation sets
        train_indexes, val_indexes = train_test_split(train_val_indexes, test_size=0.2, random_state=42)

        # Convert indexes to appropriate DataFrame indices
        train_indexes = training_labels.loc[train_indexes].index.values
        val_indexes = training_labels.loc[val_indexes].index.values
        test_indexes = training_labels.loc[test_indexes].index.values

        # Create DataLoaders
        train_loader = DataLoader(DataRetriever(train_indexes, spectrogram=spectro), batch_size=32, shuffle=True)
        val_loader = DataLoader(DataRetriever(val_indexes, spectrogram=spectro), batch_size=32, shuffle=False)
        test_loader = DataLoader(DataRetriever(test_indexes, spectrogram=spectro), batch_size=32, shuffle=False)

        # Initialize model, criterion, and optimizer
        model = Model().to('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=0.001)

        # Train and validate the model
        trainer = Trainer(model, criterion, optimizer, epochs=25)
        trainer.train(train_loader, val_loader)

        # Final evaluation on the test set
        trainer.evaluate(test_loader)
        #print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
