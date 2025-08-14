from pathlib import Path
import h5py
import numpy as np
from matplotlib import pyplot as plt
from globals import *
from loguru import logger
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

def load_feature(file_path: Path, feature_type: str = None) -> np.ndarray:
    """Loads  features from the given file path"""
    with h5py.File(file_path, 'r') as f:
        return f[feature_type][()]

def plot_feature(feature: np.ndarray, title: str = None) -> None:
    """Plots the features"""
    plt.figure(figsize=(12, 4))
    plt.imshow(feature.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Coefficients')
    plt.show()


class CremaFeatureScanner:
    """
    Scans a directory of Crema feature files to find the maximum length of a time series.
    """
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.max_length = 0
        self.file_list: list = []

        self._get_file_list()
        logger.info(f"Found {len(self.file_list)} Crema files.")
        self._scan_for_max_length()
    
    def _get_file_list(self):
        """Get list with all .h5 files' paths."""
        self.file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.directory) for f in filenames if
                          os.path.splitext(f)[1] == '.h5']

    def _scan_for_max_length(self) -> None:
        """Scans files to find the maximum length."""
        for file_path in tqdm(self.file_list):
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'crema' in f:
                        crema_len = f['crema'].shape[0]
                        self.max_length = max(self.max_length, crema_len)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        logger.info(f"Maximum Crema feature length found: {self.max_length}")

    def get_max_length(self) -> int:
        return self.max_length

#Find the maximum length of crema features in the directory which
#is also the maximum of the hpcp features


#crema_scanner = CremaFeatureScanner(CREMA_BASE_PATH)
#max_len_crema = crema_scanner.get_max_length()
    
#print(f"Maximum Crema feature length: {max_len_crema}")
#MAX_FRAMES = max_len_crema

#Function for ploting losses and accuracies
def plot_loss(epochs, losses, title = 'training- validation loss', ylabel='loss', plot_label = ['training', 'validation']):

    plt.figure()
    for i, loss in enumerate(losses):
       # print(loss)
        plt.plot(epochs, loss, label =plot_label[i])

    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.grid()
#    plt.savefig('plots/'+title+'.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{ylabel}.png', dpi=300, bbox_inches='tight')

    plt.show()



#Function to plot roc curves
def roc_c(y_true, y_pred_probs, str = 'CNN', features = 'raw '):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)

    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'(area = {roc_auc:.2f}) ')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic '+str)
    plt.grid()
    plt.legend(loc='lower right')
    #plt.savefig('plots/ROC '+str+' '+features+'.png', dpi=300, bbox_inches='tight')
    plt.savefig('ROC.png', dpi=300, bbox_inches='tight')

    plt.show()