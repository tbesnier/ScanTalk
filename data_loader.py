import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa


face = False

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertices = self.data[index]["vertices"]
        if face:
            faces = self.data[index]["faces"]
        template = self.data[index]["template"]

        if face:
            return (torch.FloatTensor(audio), torch.FloatTensor(vertices), torch.IntTensor(faces),
                torch.FloatTensor(template), file_name)
        else:
            return torch.FloatTensor(audio), torch.FloatTensor(vertices), torch.FloatTensor(template), file_name

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = args.wav_path
    vertices_path = args.vertices_path
    if face:
        faces_path = args.faces_path
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = args.template_file
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)

                key = f.replace("wav", "npy")
                data[key]["audio"] = audio_feature
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp
                vertices_path_ = os.path.join(vertices_path, f.replace("wav", "npy"))
                if face:
                    faces_path_ = os.path.join(faces_path, f.replace("wav", "npy"))
                if not os.path.exists(vertices_path_):
                    del data[key]
                else:
                    vertices = np.load(vertices_path_, allow_pickle=True)[::2, :]
                    data[key]["vertices"] = np.reshape(vertices, (vertices.shape[0], 5023, 3))
                    if face:
                        faces = np.load(faces_path_, allow_pickle=True)[::2, :]
                        data[key]["faces"] = faces


    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    splits = {'train': range(1, 41), 'val': range(1, 41), 'test': range(1, 41)}

    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    return dataset


if __name__ == "__main__":
    get_dataloaders()