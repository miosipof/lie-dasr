import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import os

import logging
logger = logging.getLogger(__name__)

# Ensure correct GPU device handling
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Set master address for communication
    os.environ["MASTER_PORT"] = "12355"  # Choose an open port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Cleanup function
def cleanup():
    dist.destroy_process_group()

class Evaluator:
    def __init__(self, model, loss_fn, val_dataset, batch_size):
        self.model = model
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.loss_fn = loss_fn
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        
    # Training function for each GPU process
    def run(self, rank, world_size):
        # Initialize the process group
        setup(rank, world_size)
    
        self.model = self.model.to(rank)
        self.encoder = self.encoder.to(rank)
        self.decoder = self.decoder.to(rank)
        
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
            
        # Loss and Optimizer
        loss_fn = self.loss_fn.to(rank)
        loss_fn.eval()
        loss_fn.perceptual_model = loss_fn.perceptual_model.to(rank)
        # loss_fn.perceptual_model.eval()
        # loss_fn.contrastive_loss.eval()
    
        # Load Dataset with Distributed Sampler
        transform = transforms.Compose([transforms.ToTensor()])
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(self.val_dataset, 
                                  batch_size=self.batch_size, 
                                  # sampler=val_sampler, 
                                  collate_fn=self.val_dataset.collate_fn,
                                  drop_last=False)
        # dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
        losses = []
    
        # Training Loop
        with torch.no_grad():
            self.model.eval()
            
            running_loss = 0.0
            running_mse = 0.0
            running_mcd = 0.0
            running_perceptual = 0.0
            running_contrastive = 0.0
            
            for batch_idx, (S_orig, output_ids_padded, target_lengths, transcription) in enumerate(val_loader):
                S_orig = S_orig.to(rank)  # Original spectrograms (B, 1, F, T)
    
                # 🔹 Step 1: Forward Pass
                E_f, E_t, encoder_hidden_state = self.encoder(S_orig)  # Extract spectral & temporal features
                E_orig = loss_fn.concat_embeds(E_f,E_t)  # Combined latent representation
    
                S_recon = self.decoder(E_f, E_t, encoder_hidden_state)  # Reconstruct spectrogram
                E_f_recon, E_t_recon, _ = self.encoder(S_recon)
                E_recon = loss_fn.concat_embeds(E_f_recon, E_t_recon)
    
                # 🔹 Step 2: Prepare contrastive loss pairs
                batch_size = S_orig.shape[0]
                perm = torch.randperm(batch_size)
                E_other = E_orig[perm]  # Shuffle embeddings to create negative samples
                
                # 🔹 Step 3: Compute Loss
                loss, mse, mcd, perceptual, contrastive = self.loss_fn(S_orig, S_recon, E_orig, E_recon, E_other)  # Autoencoder loss
    
                losses.append(
                    {
                        "loss": loss.item(),
                        "mse": mse.item(),
                        "mcd": mcd.item(),
                        "perceptual": perceptual.item(),
                        "contrastive": contrastive.item()
                    }
                )
                
                
                running_loss += loss.item()
                running_mse += mse.item()
                running_mcd +=mcd.item()
                running_perceptual += perceptual.item()
                running_contrastive += contrastive.item()
                
                if batch_idx%10 == 0:
                    msg = f"Step {batch_idx}/{len(val_loader)}: loss = {loss:.4f}, MSE {mse:.4f}, MCD {mcd:.4f}, perceptual {perceptual:.4f}, contrastive {contrastive:.4f}"
                    # logger.info(msg)
                    print(msg, flush=True)             
    
            running_loss /= len(val_loader)
            running_mse /= len(val_loader)
            running_mcd /= len(val_loader)
            running_perceptual /= len(val_loader)
            running_contrastive /= len(val_loader)
            
            print(f"Rank {rank}, Loss: {running_loss}, MSE {running_mse:.4f}, MCD {running_mcd:.4f}, perceptual {running_perceptual:.4f}, contrastive {running_contrastive:.4f}")
    
        # Cleanup process group
        cleanup()
    
        print("Evaluation Complete!")
        return losses, loss_fn