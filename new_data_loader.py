from torch.utils.data import Dataset
import torch
import numpy as np
import os
import fnmatch

class TH_Dataset(Dataset):

    def __init__(self, audio_dir, frame_dir, next_frame_dir, actor_dir, start_index, end_index):

        self.audio_dir = audio_dir
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

        audio = np.load(os.path.join(self.audio_dir, 'frame' + str(idx).zfill(6) + '.npy'))

        frame = np.load(os.path.join(self.frame_dir, 'frame' + str(idx).zfill(6) + '.npy'))
        
        next_frame = np.load(os.path.join(self.next_frame_dir, 'frame' + str(idx).zfill(6) + '.npy'))
        
        actor = np.load(os.path.join(self.actor_dir, 'frame' + str(idx).zfill(6) + '.npy'))

        audio_window = np.mean(np.array([np.load(np.load(os.path.join(self.audio_dir, 'frame' + str(i).zfill(6) + '.npy'))) for i in range(idx, idx+5)]), axis=0)

        audio = torch.Tensor(audio_window)

        frame = torch.Tensor(frame)
        
        next_frame = torch.Tensor(next_frame)
        
        actor = torch.Tensor(actor)


        sample = {'audio': audio,
                  'frame': frame,
                  'next_frame': next_frame,
                  'actor': actor}

        return sample

class TH_seq_Dataset(Dataset):

    def __init__(self, audio_dir, frame_dir, actor_dir, start_index, end_index):
        self.audio_dir = audio_dir
        self.frame_dir = frame_dir
        #self.next_frame_dir = next_frame_dir
        self.actor_dir = actor_dir
        self.start_index = start_index
        self.end_index = end_index

        print('The dataset contains: ')
        print(self.end_index - self.start_index)

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        idx = idx + self.start_index

        audio = np.load(os.path.join(self.audio_dir, 'frame' + str(idx).zfill(6) + '.npy'))

        next_audio = np.load(os.path.join(self.audio_dir, 'frame' + str(idx + 1).zfill(6) + '.npy'))

        audio_window = np.mean(np.array([np.load(os.path.join(self.audio_dir, 'frame' + str(i).zfill(6) + '.npy')) for i in range(idx-5, idx + 5)]), axis=0)

        frame = np.load(os.path.join(self.frame_dir, 'frame' + str(idx).zfill(6) + '.npy'))

        next_frame = np.load(os.path.join(self.frame_dir, 'frame' + str(idx+1).zfill(6) + '.npy'))

        actor = np.load(os.path.join(self.actor_dir, 'frame' + str(idx).zfill(6) + '.npy'))

        audio, next_audio = torch.Tensor(audio_window), torch.Tensor(next_audio)

        frame = torch.Tensor(frame)

        next_frame = torch.Tensor(next_frame)

        actor = torch.Tensor(actor)

        sample = {'audio': audio,
                  'next_audio': next_audio,
                  'frame': frame,
                  'next_frame': next_frame,
                  'actor': actor}

        return sample


class TH_pretrain_spiral_Dataset(Dataset):

    def __init__(self, frame_dir, actor_dir, start_index, end_index):

        self.frame_dir = frame_dir
        self.actor_dir = actor_dir
        self.start_index = start_index
        self.end_index = end_index

        print('The dataset contains: ')
        print(self.end_index - self.start_index)

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        idx = idx + self.start_index

        frame = np.load(os.path.join(self.frame_dir, 'frame' + str(idx).zfill(6) + '.npy'))

        actor = np.load(os.path.join(self.actor_dir, 'frame' + str(idx).zfill(6) + '.npy'))

        frame = torch.Tensor(frame)

        actor = torch.Tensor(actor)

        sample = {'frame': frame,
                  'actor': actor}

        return sample