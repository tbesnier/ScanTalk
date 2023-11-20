import numpy as np
from transformers import Wav2Vec2Processor
import librosa
import os
import torch
from wav2vec import Wav2Vec2Model
import pickle
from scipy.io import wavfile


data_path = '../datasets/VOCA_training/vertices_npy'
audio_path = '../datasets/VOCA_training/wav'

with open('../datasets/VOCA_training/templates.pkl', 'rb') as fin:
    templates = pickle.load(fin, encoding='latin1')

audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# wav2vec 2.0 weights initialization
audio_encoder.feature_extractor._freeze_parameters()

j = 0
for audio in os.listdir(audio_path):
    audios = []
    frame = []
    frame_1 = []
    frame_2 = []
    actor = str(audio[:24])
    template = templates[actor]
    sentence = audio[:-4] + '.npy'
    print(os.path.join(data_path, sentence))
    if os.path.exists(os.path.join(data_path, sentence)):
        vertices = np.load(os.path.join(data_path, sentence))
        vertices = np.reshape(vertices, (len(vertices), 5023, 3))
        print(vertices)
        speech_array, sampling_rate = librosa.load(os.path.join(audio_path, audio), sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature)
        hidden_states = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state
        hidden_states = hidden_states.detach().numpy().squeeze(0)
        os.makedirs('../Data/VOCA/Consecutive_Dataset/Frame', exist_ok=True)
        #os.makedirs('../Data/VOCA/Consecutive_Dataset/Next_Frame', exist_ok=True)
        os.makedirs('../Data/VOCA/Consecutive_Dataset/Audio_Snippet', exist_ok=True)
        os.makedirs('../Data/VOCA/Consecutive_Dataset/Actor', exist_ok=True)
        for i in range(len(vertices)-1):
            print(j)       
            np.save('../Data/VOCA/Consecutive_Dataset/Frame/frame' + str(j).zfill(6) + '.npy', vertices[i])
            #np.save('../Data/VOCA/Consecutive_Dataset/Next_Frame/frame' + str(j).zfill(6) + '.npy', vertices[i+1])
            np.save('../Data/VOCA/Consecutive_Dataset/Audio_Snippet/frame' + str(j).zfill(6) + '.npy', hidden_states[i])
            np.save('../Data/VOCA/Consecutive_Dataset/Actor/frame' + str(j).zfill(6) + '.npy', template)
            j+=1
        
        
        
