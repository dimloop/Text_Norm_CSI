#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Set
from globals import *
import random
import json

class GeneratePairs:
    "Class to generate pairs of similar and dissimilar songs based on metadata."
    def __init__(self, num_pairs: int):
        with open(METADATA_PATH, 'r') as f:
            self.metadata = json.load(f)
        self.num_pairs = num_pairs
        
    def get_pairs(self) -> List[Tuple[str, str, str, str, int]]:
        return self.all_training_pairs

    def __len__(self) -> int:
        return len(self.all_training_pairs)
    
    def get_songs_with_covers(self) -> List[str]:
        return [wid for wid, performances in self.metadata.items() if len(performances) > 1]
    
    def sort_pair(self, pid1: str, pid2: str) -> Tuple[str, str]:
        """Sorts a pair of PIDs alphabetically to ensure consistency."""
        return tuple(sorted((pid1, pid2)))

    def create_datapoints(self, songs_with_covers: List[str]) -> List[Tuple[str, str, int]]:
        """Generates a list of similar and dissimilar song pairs, ensuring no duplicates."""
        all_pairs_list = []
        generated_pairs: Set[Tuple[str, str]] = set()
        num_similar_pairs = self.num_pairs // 2
        #Generate similar pairs (label 0) 
        while len(generated_pairs) < num_similar_pairs:
            random_work_id = random.choice(songs_with_covers)
            pid1, pid2 = random.sample(list(self.metadata[random_work_id].keys()), 2)
        
            sorted_pair = self.sort_pair(pid1, pid2)
            if sorted_pair not in generated_pairs:
                generated_pairs.add(sorted_pair)
                all_pairs_list.append((random_work_id, pid1, random_work_id, pid2, 0))
    
        #Generate dissimilar pairs (label 1) 
        all_pids = list({pid for wid in self.metadata for pid in self.metadata[wid].keys()})
    
        while len(generated_pairs) < self.num_pairs:
            pid1 = random.choice(all_pids)
            pid2 = random.choice(all_pids)
        
            if pid1 == pid2:
                continue
            
            wid1 = next(w for w, p in self.metadata.items() if pid1 in p)
            wid2 = next(w for w, p in self.metadata.items() if pid2 in p)
        
            if wid1 != wid2:
                sorted_pair = self.sort_pair(pid1, pid2)
                if sorted_pair not in generated_pairs:
                    generated_pairs.add(sorted_pair)
                    all_pairs_list.append((wid1, pid1, wid2, pid2, 1))
    
        random.shuffle(all_pairs_list)
    
        return all_pairs_list
    
    def generate_pairs(self) -> List[Tuple[str, str, str, str, int]]:
        """Generates pairs of songs with their respective labels."""
        songs_with_covers = self.get_songs_with_covers()
        self.all_training_pairs = self.create_datapoints(songs_with_covers)
        return self.all_training_pairs





## Dataset Class
class CoverSongDataset(Dataset):
    """
    A PyTorch Dataset for loading similar and dissimilar song pairs.
    It loads HPCP and Crema features from HDF5 files based on song IDs.
    It can be used to create single-channel or multi-channel features.
    """
    def __init__(self, pairs: List[Tuple[str, str, int]], features: List[str], max_frames: int = 2000, conv1d: bool = True):
        """
        Args:
            pairs (List[Tuple[str, str, int]]): A list of (work_id_A, song_id_A, work_id_B, song_id_B, label) tuples.
            features (List[str]): A list of feature types to load (e.g., ['hpcp', 'crema']).
            hpcp_dir (Path): The directory containing the HPCP feature files.
            crema_dir (Path): The directory containing the Crema feature files.
            max_frames (int): The maximum number of frames to pad all features to.
        """
        self.pairs = pairs
        self.features = features  #Store the feature types to load
        self.hpcp_dir = HPCP_BASE_PATH
        self.crema_dir = CREMA_BASE_PATH
        self.max_frames = max_frames#2000#MAX_FRAMES #Store the max_frames value
        self.downsample = True  # Set to True if you want to downsample features
        self.pad = True  # Set to True if you want to pad features to max_frames
        self.conv1d = conv1d  # Set to True if you want to use 1D convolution
    
    def __len__(self):
        return len(self.pairs)

    def load_feature(self, file_path: Path, feature_type: str) -> np.ndarray:
        """Loads features from the given file path."""
        with h5py.File(file_path, 'r') as f:
            feature = f[feature_type][()]
            

            # 1) resample to exactly self.max_frames
            #feature = self.resample_feature(feature, num_frames=self.max_frames)

            # 2) normalize per frame (L2) AFTER resampling
            #feature = self.normalize_feature(feature)
            
            #feature =self.normalize_feature(feature)
            
            feature = self.downsample_feature(feature, num_frames=self.max_frames) 
            feature = self.pad_feature(feature, num_frames=self.max_frames)
            feature = self.trim_feature(feature, num_frames=self.max_frames)
            
            
            #if self.downsample:
            #    feature = self.downsample_feature(feature, num_frames=self.max_frames)
            #if self.pad:
            #    feature = self.pad_feature(feature, num_frames=self.max_frames)
            
            return feature


    def resample_feature(self, feature: np.ndarray, num_frames: int = 2000) -> np.ndarray:
        """
        Time-resample to exactly num_frames via linear interpolation.
        feature: [T_in, F] -> [num_frames, F]
        """
        T_in, F = feature.shape
        if T_in == num_frames:
            return feature

        # old/new normalized time axes (0..1)
        x_old = np.linspace(0.0, 1.0, T_in, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num_frames, endpoint=False)

        out = np.empty((num_frames, F), dtype=feature.dtype)
        # np.interp is 1D, so loop over feature dims (F is small: 12/24)
        for j in range(F):
            out[:, j] = np.interp(x_new, x_old, feature[:, j])
        return out
    
    
    def normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """Normalizes the feature using normalization."""
        
        # feature shape: (frames, coefficients)
        return normalize(feature, axis=1, norm='l2')  # Normalize along the time axis
    
    def downsample_feature(self, feature: np.ndarray, num_frames: int = 2000) -> np.ndarray:
        """Downsamples the feature to a target number of frames."""
        
        if len(feature) > num_frames:
            downsample_factor = len(feature) // num_frames
            return feature[::downsample_factor]
        else:
            return feature
    
    def pad_feature(self, feature: np.ndarray, num_frames: int = 2000) -> np.ndarray:
        """Pads the feature to a target number of frames."""
        
        if len(feature) < num_frames:
            pad_width = ((0, num_frames - len(feature)), (0, 0))
            return np.pad(feature, pad_width, mode='constant')
        else:            
            return feature
    
    def trim_feature(self, feature: np.ndarray, num_frames: int = 2000) -> np.ndarray:
        """Trims the feature to a target number of frames."""
        
        if len(feature) > num_frames:
            return feature[:num_frames]
        else:
            return feature
        
    def __getitem__(self, index):
        work_a_id, song_a_id, work_b_id, song_b_id, label = self.pairs[index]

        if self.features is None or len(self.features) == 0:
            raise ValueError("No features specified to load.")  
        elif len(self.features) > 2:
            raise ValueError("Only 'hpcp' and 'crema' features are supported for loading.")
        
        elif len(self.features) == 1 and 'hpcp' in self.features:
            # Load only HPCP feature
            hpcp_path_a = self.hpcp_dir / f"{work_a_id}_hpcp" / f"{song_a_id}_hpcp.h5"
            hpcp_path_b = self.hpcp_dir / f"{work_b_id}_hpcp" / f"{song_b_id}_hpcp.h5"
            feature_a = self.load_feature(hpcp_path_a, 'hpcp')
            feature_b = self.load_feature(hpcp_path_b, 'hpcp')
            return torch.from_numpy(feature_a).float(), torch.tensor(feature_b, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        elif len(self.features) == 1 and 'crema' in self.features:
            # Load only Crema feature
            crema_path_a = self.crema_dir / f"{work_a_id}_crema" / f"{song_a_id}_crema.h5"
            crema_path_b = self.crema_dir / f"{work_b_id}_crema" / f"{song_b_id}_crema.h5"
            feature_a = self.load_feature(crema_path_a, 'crema')
            feature_b = self.load_feature(crema_path_b, 'crema')
            return torch.from_numpy(feature_a).float(), torch.tensor(feature_b, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        elif len(self.features) == 2 and 'hpcp' in self.features and 'crema' in self.features:
            # Load both HPCP and Crema features
            hpcp_path_a = self.hpcp_dir / f"{work_a_id}_hpcp" / f"{song_a_id}_hpcp.h5"
            hpcp_path_b = self.hpcp_dir / f"{work_b_id}_hpcp" / f"{song_b_id}_hpcp.h5"
            crema_path_a = self.crema_dir / f"{work_a_id}_crema" / f"{song_a_id}_crema.h5"
            crema_path_b = self.crema_dir / f"{work_b_id}_crema" / f"{song_b_id}_crema.h5"

            hpcp_f_a = self.load_feature(hpcp_path_a, 'hpcp')
            crema_f_a = self.load_feature(crema_path_a, 'crema')
            hpcp_f_b = self.load_feature(hpcp_path_b, 'hpcp')
            crema_f_b = self.load_feature(crema_path_b, 'crema') 
            feature_a = torch.from_numpy(np.stack([hpcp_f_a, crema_f_a], axis=0)).float()
            feature_b = torch.from_numpy(np.stack([hpcp_f_b, crema_f_b], axis=0)).float()
            label = torch.tensor(label, dtype=torch.float32) 
            if self.conv1d: 
                
                C, T, F = feature_a.shape
                return feature_a.reshape(C*F, T).permute(1,0), \
                       feature_b.reshape(C*F, T).permute(1,0), \
                       label
            else:   
                return feature_a, \
                       feature_b, \
                       label
        
            
        #print(hpcp_f_a, crema_f_a.shape,hpcp_f_b.shape, crema_f_b.shape)
        
        #plt.imshow(hpcp_f_a.T, aspect='auto', origin='lower')
        #plt.colorbar()
        #plt.show()
       
        #plt.imshow(self.normalize_feature(hpcp_f_a).T, aspect='auto', origin='lower')
        #plt.colorbar()
        #plt.show()

        
       
        
        #return #feature_a, feature_b, torch.tensor(label, dtype=torch.float32)

#Example usage
#data = CoverSongDataset(all_training_pairs, ['hpcp','crema'])
#data[0]

from torch.utils.data import DataLoader
from typing import Dict
def create_dataloaders(dataset: CoverSongDataset, batch_size: int = 32, validation_split: float = 0.2, test_split: float = 0.1) -> Dict[str, DataLoader]:
    """
    Creates training and validation DataLoaders from the dataset.
    
    Args:
        dataset (CoverSongDataset): The dataset to create DataLoaders from.
        batch_size (int): Batch size for the DataLoaders.
        validation_split (float): Fraction of the dataset to use for validation.
        
    Returns:
        Dict[str, DataLoader]: Dictionary containing 'train' and 'validation' DataLoaders.
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    n_test = int(np.floor(test_split * dataset_size))
    n_val  = int(np.floor(validation_split * dataset_size))
    n_train = dataset_size - n_test - n_val

    test_indices = indices[:n_test]
    val_indices  = indices[n_test:n_test + n_val]
    train_indices = indices[n_test+n_val:]
    print(n_test, n_val, n_train, dataset_size)
    np.random.shuffle(indices)
    
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(dataset, val_indices)
    test_dataset  = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    print(len(train_loader))
    return {"train": train_loader, "validation": val_loader, "test": test_loader}