from torch.utils.data import Dataset
import torch
import numpy as np
import os
import fnmatch

class TH_Dataset(Dataset):

    def __init__(self, audio_dir_0, audio_dir_1, audio_dir_2, frame_dir, next_frame_dir, actor_dir, start_index, end_index):

        self.audio_dir_0 = audio_dir_0
        self.audio_dir_1 = audio_dir_1
        self.audio_dir_2 = audio_dir_2
        self.frame_dir = frame_dir
        self.next_frame_dir = next_frame_dir
        self.actor_dir = actor_dir
        self.start_index = start_index
        self.end_index = end_index

        print('The dataset contains: ')
        print(self.end_index - self.start_index)

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        
        idx = idx + self.start_index

        audio_0 = np.load(os.path.join(self.audio_dir_0, 'frame' + str(idx).zfill(6) + '.npy'))
        
        audio_1 = np.load(os.path.join(self.audio_dir_1, 'frame' + str(idx).zfill(6) + '.npy'))
        
        audio_2 = np.load(os.path.join(self.audio_dir_2, 'frame' + str(idx).zfill(6) + '.npy'))
        
        frame = np.load(os.path.join(self.frame_dir, 'frame' + str(idx).zfill(6) + '.npy'))
        
        next_frame = np.load(os.path.join(self.next_frame_dir, 'frame' + str(idx).zfill(6) + '.npy'))
        
        actor = np.load(os.path.join(self.actor_dir, 'frame' + str(idx).zfill(6) + '.npy'))
        
        audio_0 = torch.Tensor(audio_0)
        
        audio_1 = torch.Tensor(audio_1)
        
        audio_2 = torch.Tensor(audio_2)
        
        frame = torch.Tensor(frame)
        
        next_frame = torch.Tensor(next_frame)
        
        actor = torch.Tensor(actor)
        
        sample = {'audio_0': audio_0, 
                  'audio_1': audio_1, 
                  'audio_2': audio_2, 
                  'frame': frame, 
                  'next_frame': next_frame,
                  'actor': actor}

        return sample

