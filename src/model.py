import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lie import apply_random_curriculum_transform

class Processor(nn.Module):
    def __init__(self,asr_model,device):
        super(Processor, self).__init__()
        self.asr_model = asr_model
        self.device = device

    def pad_spectrogram(self, spectrogram, target_length=512):
        """Pads or truncates spectrogram to a fixed time dimension."""
        # spectrogram = spectrogram.permute(0,1,3,2)
        _, _, T = spectrogram.shape  # Shape (batch, freq, time)
        
        if T < target_length:
            pad_amount = target_length - T
            spectrogram = F.pad(spectrogram, (0, pad_amount), mode='constant', value=0)  # Pad on the right
        elif T > target_length:
            spectrogram = spectrogram[:, :, :target_length]  # Truncate
            
        return spectrogram

    def preprocess_nemo(self, wf_batch):
        wf_lengths = [torch.tensor(len(wf),dtype=torch.int64) for wf in wf_batch]
        wf_lengths = torch.stack(wf_lengths, dim=0).to(self.device)

        wf_batch = [torch.tensor(wf).to(self.device) for wf in wf_batch]
        wf_batch = torch.stack(wf_batch,dim=0)
        
        S, L = self.asr_model.preprocessor.get_features(wf_batch,wf_lengths)
        S = self.pad_spectrogram(S,target_length=512).float()
        return S
        
    def preprocess_nemo_single(self, wf):

        wf_lengths = torch.tensor(len(wf)).unsqueeze(0).to(self.device) # add batch dim
        wf_batch = torch.tensor(wf).unsqueeze(0).to(self.device) # add batch dim
        
        S, L = self.asr_model.preprocessor.get_features(wf_batch,wf_lengths)
        S = self.pad_spectrogram(S,target_length=512).float()
        return S
        
    def encode_nemo(self, S):
        S = S.squeeze(1).float()
        
        input_lengths = torch.full(
            size=(S.size(0),),  # batch size
            fill_value=S.size(-1),  # T
            dtype=torch.int64
        ).to(S.device)    
        
        encoder_out, encoded_len =  self.asr_model(processed_signal=S, processed_signal_length=input_lengths)
        return encoder_out, encoded_len

        
    def preprocess_whisper(self, x):
        # audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self.asr_model.feature_extractor(
            x["audio"]["array"],
            sampling_rate=self.asr_model.feature_extractor.sampling_rate,
            # max_length=int(feature_extractor.sampling_rate * max_duration),
            # truncation=True,
        )
    
        x["in_spectrogram"] = torch.tensor(inputs['input_features'],dtype=torch.float32).squeeze(0)
    
        return x
    
        
    # def encode_whisper(self, spectrogram):
    #     spectrogram = spectrogram.squeeze(1)
    #     spectrogram = self.pad_spectrogram(spectrogram, target_length=3000)
    #     with torch.no_grad():
    #         features = self.whisper_model.encoder(spectrogram).last_hidden_state
    
    #     return features


    def forward(self, S):
        S = S.squeeze(1).float()
        
        input_lengths = torch.full(
            size=(S.size(0),),  # batch size
            fill_value=S.size(-1),  # T
            dtype=torch.int64
        ).to(S.device)    
        
        encoder_out, encoded_len =  self.asr_model(processed_signal=S, processed_signal_length=input_lengths)
        
        return encoder_out



class SpectrogramAugmentDataset(Dataset):
    def __init__(self, processor, data, max_epochs):
        """
        data: a list of dicts with 'in_spectrogram': Tensor [F, T]
        max_epochs: total training epochs (for curriculum)
        transform_epoch_getter: function returning current epoch
        """
        self.processor = processor
        self.data = data
        self.max_epochs = max_epochs
        self.train = True        
        self.epoch = 0        
        self.epsilon_dict = {
                                't_stretch': 0,
                                'f_stretch': 0,
                                'warp_2d': 0,
                                'amplitude': 0,
                                'phase': 0
                            }
        
        self.random_transform_ids = [0,1,2,3] # [0,1,2]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        wf = self.data[idx]['audio']['array']
        spec = self.processor.preprocess_nemo_single(wf).squeeze(0)

        if self.train:
            distorted, fields, epsilon_dict, selected_transform = apply_random_curriculum_transform(spec, 
                                                                             self.epoch, 
                                                                             self.max_epochs, 
                                                                             device=spec.device,
                                                                             epsilon_dict = self.epsilon_dict,
                                                                             transform_ids = self.random_transform_ids)
        else:
            # distorted, epsilon, selected = apply_random_curriculum_transform(spec, self.max_epochs, self.max_epochs)
            distorted, fields, epsilon_dict, selected_transform = apply_random_curriculum_transform(spec, 
                                                                             self.epoch, 
                                                                             self.max_epochs, 
                                                                             device=spec.device,
                                                                             epsilon_dict = self.epsilon_dict,
                                                                             transform_ids = self.random_transform_ids)
            
        return distorted, spec, fields, epsilon_dict, selected_transform

