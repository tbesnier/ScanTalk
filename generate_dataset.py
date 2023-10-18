import numpy as np
from transformers import Wav2Vec2Processor
import librosa
import os
import torch
from wavlm import Wav2Vec2Model
import pickle


data_path = '/home/federico/Scrivania/TH/S2L/vocaset/vertices_npy'
audio_path = '/home/federico/Scrivania/TH/S2L/vocaset/wav'

with open('/home/federico/Scrivania/TH/S2L/vocaset/templates.pkl', 'rb') as fin:
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
    if os.path.exists(os.path.join(data_path, sentence)):
        vertices = np.load(os.path.join(data_path, sentence))
        vertices = np.reshape(vertices, (len(vertices), 5023, 3))
        speech_array, sampling_rate = librosa.load(os.path.join(audio_path, audio), sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature)
        hidden_states = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state
        hidden_states = hidden_states.detach().numpy().squeeze(0)
        
        for i in range(len(vertices)-1):
            print(j)       
            np.save('/home/federico/Scrivania/ScanTalk/ScanTalk/Consecutive_Dataset/Frame/frame' + str(j).zfill(6) + '.npy', vertices[i])
            np.save('/home/federico/Scrivania/ScanTalk/ScanTalk/Consecutive_Dataset/Next_Frame/frame' + str(j).zfill(6) + '.npy', vertices[i+1])
            np.save('/home/federico/Scrivania/ScanTalk/ScanTalk/Consecutive_Dataset/Audio_Snippet/frame' + str(j).zfill(6) + '.npy', hidden_states[i+1])
            np.save('/home/federico/Scrivania/ScanTalk/ScanTalk/Consecutive_Dataset/Actor/frame' + str(j).zfill(6) + '.npy', template)
            j+=1
        
        
        
