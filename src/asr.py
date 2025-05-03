import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn


class SpeechDataset(Dataset):
    def __init__(self, dataset, asr_model, text_column_name="transcription"):
        self.dataset = dataset
        self.text_column_name = text_column_name
        self.asr_model = asr_model

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        spectrogram = self.dataset[idx]["in_spectrogram"]
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).squeeze(0) # For IT Commonvoice STS dataset (1, 1, 3000, 80)
        # spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0) # For BB ASR dataset (3000, 80)
        # Needed shape: (1, 3000, 80)
                
        text = self.dataset[idx][self.text_column_name]
        target_token_ids = [self.asr_model.tokenizer.text_to_ids(t) for t in text]
        # target_token_ids = [t for tokens in target_token_ids for t in tokens] # flatten n-token combinations

        return spectrogram, target_token_ids

    def collate_fn(self,batch):

        spectrograms, tokens_batched = zip(*batch)
    
        tokens_flattened = []
        for token_seq in tokens_batched:
            token_seq_flattened = [torch.tensor(i, dtype=torch.int64) for ids in token_seq for i in ids]
            tokens_flattened.append(token_seq_flattened)
        
        target_lengths = torch.tensor([len(token_seq_flattened) for token_seq_flattened in tokens_flattened], dtype=torch.int64)

        pad_token_id = 0
        max_len = max(target_lengths)
        tokens_padded = []
        for token_seq in tokens_flattened:
            token_seq_padded = token_seq + [pad_token_id] * (max_len - len(token_seq))
            tokens_padded.append(torch.tensor(token_seq_padded, dtype=torch.int64))
        
        targets = torch.stack(tokens_padded,dim=0)
        
        spectrograms = torch.stack(spectrograms,dim=0)

        return spectrograms, targets, target_lengths

