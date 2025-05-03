import logging
logging.disable(logging.CRITICAL)


from datasets import load_dataset, Audio, DatasetDict, Dataset, load_from_disk, IterableDatasetDict, interleave_datasets, concatenate_datasets
from datetime import timedelta

from transformers import WhisperModel, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig, WhisperTokenizer, logging, BitsAndBytesConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from huggingface_hub import notebook_login
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import multiprocess as mp
import gc
import time
import numpy as np
from scipy import signal
import librosa
import evaluate
import jiwer
import os
import numba
import json
from numba import jit, cuda
import random
import torchaudio
from torch.nn import ZeroPad2d
from torch.utils.data import Dataset, DataLoader
import nemo.collections.asr as nemo_asr
import torch.nn.attention
import torch.multiprocessing as mp
from multiprocessing import Queue
from torch.multiprocessing import SimpleQueue
from nemo.collections.asr.models import EncDecRNNTModel
import torch.nn as nn
from collections import OrderedDict

from src.lie_trainer import LieTrainer
from src.model import Processor, SpectrogramAugmentDataset
from src.evaluation import Evaluator
from src.asr import SpeechDataset

sampling_rate = 16000
os.environ['KMP_DUPLICATE_LIB_OK']='True'
logging.set_verbosity_warning()
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.version.cuda)





class LieFieldPredictor(nn.Module):
    def __init__(self, encoder_name='efficientnet-b0', pretrained=True, in_channels=1, out_channels=5):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None  # No activation here — leave that to post-processing if needed
        )

    def forward(self, x):
        """
        Input: x of shape [B, 1, F, T] (e.g. [B, 1, 80, 512])
        Output: y of shape [B, 5, F, T] — one field per channel
        """
        return self.unet(x)


datasets = IterableDatasetDict()

############################### CommonVoice EN STS ###############################
datasets = load_dataset("miosipof/StS_Nemo_CommonVoice")
ds_len = len(datasets["train"])
datasets["val"] = datasets["train"].select(range(int(ds_len*0.1))).select(range(2000))
datasets["train"] = datasets["train"].select(range(int(ds_len*0.1), ds_len-1)).select(range(20000))


# Ensure correct GPU device handling
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Set master address for communication
    os.environ["MASTER_PORT"] = "12358"  # Choose an open port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)

# Cleanup function
def cleanup():
    dist.destroy_process_group()

    


def create_asr_model(rank):
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
    asr_model.cfg.joint['fuse_loss_wer'] = False

    asr_model.joint = EncDecRNNTModel.from_config_dict(asr_model.cfg.joint)

    asr_model.spec_augmentation = None  # 🔕 disables SpecAugment 
    # asr_model.preprocessor = None  # Disable built-in preprocessing  
    
    return asr_model




def create_model(rank):

    device = torch.device(f"cuda:{rank}")

    model = smp.Unet(
        encoder_name="resnext50_32x4d",       # or "efficientnet-b0" or "resnet-34"
        encoder_weights="imagenet",    # pretrained!
        in_channels=1,                 # spectrogram = single channel
        classes=5                      # your scalar fields
    )
        

    # weights = torch.load("Lie_ResNet_ssb_v3_10_epochs.pth")
    # state_dict = weights['model_state_dict']
    # new_state_dict = []
        
    # for k,v in state_dict.items():
    #     k = k.replace('module.','')
    #     new_state_dict.append((k,v))
    
    # new_state_dict = OrderedDict(new_state_dict)
    # model.load_state_dict(new_state_dict)

    print(f"ResNet parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model


import torch.distributed as dist

def train_worker(rank, world_size, model_name, model_args, loss_args, train_dataset, val_dataset, batch_size, lr, num_epochs, result_queue):
    # Set up distributed training here
    torch.cuda.set_device(rank)
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    setup(rank, world_size)

    # Create model inside the worker
    asr_model = create_asr_model(rank)
    model = create_model(rank)

    processor = Processor(asr_model=asr_model, device=rank)   

    train_dataset = SpectrogramAugmentDataset(processor, train_dataset, num_epochs)
    val_dataset = SpectrogramAugmentDataset(processor, val_dataset, num_epochs)

    loss_weights = {
        "hat": 1.0,
        "v": 1.0,
        "cos": 1.0,
        "spec": 1.0,
        "smooth": 1.0
    }

    epsilon_base, epsilon_plateau, epsilon_max = 0.1, 0.5, 1.0

    trainer = LieTrainer(
        model_name, model,
        train_dataset, val_dataset,
        batch_size, lr, num_epochs,
        loss_weights, epsilon_base, epsilon_plateau, epsilon_max
    )

    try:
        result = trainer.train(rank, world_size)
        result_queue.put((rank, result))
        dist.barrier()
    finally:
        cleanup()    
    

    return result




def start_training(model_name, model_args, loss_args, train_dataset, val_dataset, lr=0.001, num_epochs=5, batch_size=32):
    world_size = torch.cuda.device_count() # Number of available GPUs
    processes = []
    result_queue = SimpleQueue() # Queue()
    
    for rank in range(world_size):
        p = mp.Process(target=train_worker, args=(
            rank, world_size, model_name, model_args, loss_args, train_dataset, val_dataset, batch_size, lr, num_epochs, result_queue
        ))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    # Gather results
    all_results = [None] * world_size
    while not result_queue.empty():
        rank, result = result_queue.get()
        all_results[rank] = result

    return all_results

    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.cuda.empty_cache()
    
    batch_size=16
    lr=3e-5
    num_epochs=20

    model_name = "Lie_ResNet_ssb_v5"
        
    result = start_training(model_name, None, None, 
                   datasets["train"].select(range(10000)), 
                   datasets["val"].select(range(500)), 
                   lr=lr, num_epochs=num_epochs, batch_size=batch_size)

    print(result)

    with open(model_name+'_results.json', 'w') as fp:
        json.dump(result, fp)
