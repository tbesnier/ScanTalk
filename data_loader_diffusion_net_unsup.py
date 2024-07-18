import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
import sys
import trimesh

sys.path.append('./model/diffusion-net/src')
import model.diffusion_net as diffusion_net



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
        template = self.data[index]["template"]
        mass = self.data[index]["mass"]
        L = self.data[index]["L"]
        evals = self.data[index]["evals"]
        evecs = self.data[index]["evecs"]
        gradX = self.data[index]["gradX"]
        gradY = self.data[index]["gradY"]
        faces_template = self.data[index]["faces_template"]
        #faces = self.data[index]["faces"]
        dataset = self.data[index]["dataset"]
        return torch.FloatTensor(audio), torch.FloatTensor(vertices), torch.FloatTensor(template), torch.FloatTensor(np.array(mass)).float(), L.float(), torch.FloatTensor(np.array(evals)), torch.FloatTensor(np.array(evecs)), gradX.float(), gradY.float(), file_name, faces_template.float(), dataset

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
    #faces_path = args.faces_path
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    #reference = trimesh.load('./template/flame_model/FLAME_sample.ply',
    #                         process=False)
    #template_tri = reference.faces
    #template_file = args.template_file
    #with open(template_file, 'rb') as fin:
    #    templates = pickle.load(fin, encoding='latin1')

    subject_id_list = []
    mass_dict = {}
    L_dict = {}
    evals_dict = {}
    evecs_dict = {}
    gradX_dict = {}
    gradY_dict = {}

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):# and f[3] != 'e':
                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = audio_feature
                data[key]["name"] = f

                subject_id = "_".join(key.split("_")[:-1])
                #if subject_id in subjects_dict["train"] or subject_id in subjects_dict["val"] or subject_id in subjects_dict["test"]:
                template_mesh = trimesh.load(os.path.join(args.template_dir, subject_id + '.ply'), process=False)
                temp = template_mesh.vertices
                data[key]["template"] = temp
                #temp = templates[subject_id]
                #data[key]["template"] = temp
                vertices_path_ = os.path.join(vertices_path, f.replace("wav", "npy"))
                #faces_path_ = os.path.join(faces_path, f.replace("wav", "npy"))
                if subject_id not in subject_id_list:
                    subject_id_list.append(subject_id)
                    frame, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
                        torch.tensor(temp), faces=torch.tensor(template_mesh.faces), k_eig=args.k_eig)
                    mass_dict[subject_id] = mass
                    L_dict[subject_id] = L
                    evals_dict[subject_id] = evals
                    evecs_dict[subject_id] = evecs
                    gradX_dict[subject_id] = gradX
                    gradY_dict[subject_id] = gradY

                data[key]["mass"] = mass_dict[subject_id]
                data[key]["L"] = L_dict[subject_id]
                data[key]["evals"] = evals_dict[subject_id]
                data[key]["evecs"] = evecs_dict[subject_id]
                data[key]["gradX"] = gradX_dict[subject_id]
                data[key]["gradY"] = gradY_dict[subject_id]
                data[key]["faces_template"] = torch.tensor(template_mesh.faces)
                data[key]["dataset"] = "vocaset"

                if not os.path.exists(vertices_path_):
                    del data[key]
                else:
                    vertices = np.load(vertices_path_, allow_pickle=True)[::2, :, :]
                    #print(vertices.shape)
                    data[key]["vertices"] = vertices #np.reshape(vertices, (vertices.shape[0], vertices.shape[1]//3, 3))
                    #data[key]["vertices"] = vertices
                    #faces = np.load(faces_path_, allow_pickle=True)[::2, :]
                    #data[key]["faces"] = np.reshape(faces, (faces.shape[0], faces.shape[1]//3, 3))

    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        if subject_id in subjects_dict["train"]: #and sentence_id in splits['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:# and sentence_id in splits['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:# and sentence_id in splits['test']:
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

