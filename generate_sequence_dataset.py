import numpy as np
from transformers import AutoProcessor
import librosa
import os
import torch
from wavlm import WavLMModel
import pickle


data_path = '/home/federico/Scrivania/TH/S2L/vocaset/vertices_npy'
audio_path = '/home/federico/Scrivania/TH/S2L/vocaset/wav'

with open('/home/federico/Scrivania/TH/S2L/vocaset/templates.pkl', 'rb') as fin:
    templates = pickle.load(fin, encoding='latin1')

audio_encoder = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")

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
    print(j)
    if os.path.exists(os.path.join(data_path, sentence)):
        vertices = np.load(os.path.join(data_path, sentence))
        vertices = np.reshape(vertices, (len(vertices), 5023, 3))
        speech_array, sampling_rate = librosa.load(os.path.join(audio_path, audio), sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature)
        hidden_states = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state
        hidden_states = hidden_states.detach().numpy().squeeze(0)
        
        np.save('/home/federico/Scrivania/ScanTalk/Sequence_Dataset_WavLM/Sequences/seq' + str(j).zfill(3) + '.npy', vertices)
        np.save('/home/federico/Scrivania/ScanTalk/Sequence_Dataset_WavLM/Audios/seq' + str(j).zfill(3) + '.npy', hidden_states)
        np.save('/home/federico/Scrivania/ScanTalk/Sequence_Dataset_WavLM/Templates/seq' + str(j).zfill(3) + '.npy', template)
        j+=1
        
        
        
