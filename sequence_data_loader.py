from torch.utils.data import Dataset
import torch
import numpy as np
import os
import fnmatch

class TH_Dataset(Dataset):

    def __init__(self, audio_dir, sequence_dir, template_dir, start_index, end_index):

        self.audio_dir = audio_dir
        self.sequence_dir = sequence_dir
        self.template_dir = template_dir
        self.start_index = start_index
        self.end_index = end_index

        print('The dataset contains: ')
        print(self.end_index - self.start_index)

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        
        idx = idx + self.start_index

        audio = np.load(os.path.join(self.audio_dir, 'seq' + str(idx).zfill(3) + '.npy'))
        
        sequence = np.load(os.path.join(self.sequence_dir, 'seq' + str(idx).zfill(3) + '.npy'))
        
        actor = np.load(os.path.join(self.template_dir, 'seq' + str(idx).zfill(3) + '.npy'))
        
        audio = torch.Tensor(audio)
        
        sequence = torch.Tensor(sequence)
        
        actor = torch.Tensor(actor)
        
        sample = {'audio': audio,
                  'sequence': sequence,
                  'actor': actor}

        return sample

