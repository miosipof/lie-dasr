import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import torch.nn.functional as F


import random
import json
import numpy as np
import os
import gc

from src.losses import spatial_smoothness_loss, general_sparsity_loss, sparsity_loss, cosine_field_loss, phi4_loss_fn
from src.lie import localized_smooth_field_1d, localized_smooth_field_2d, generate_lie_generator_fields, apply_transformation
from src.utils import pad_spectrogram, imagenet_normalize_1ch, normalize_spectrogram, denormalize_spectrogram
from src.utils import normalize_field, denormalize_field, normalize_fields, denormalize_fields
from src.lie import apply_inverse_transform, batch_grid_warp


from pprint import pprint as pp

import logging
logger = logging.getLogger(__name__)




class LieTrainer:
    def __init__(self, model_name, model, 
                 train_dataset, val_dataset, 
                 batch_size, lr, num_epochs, 
                 loss_weights, epsilon_base,epsilon_plateau,epsilon_max):
        
        self.model_name = model_name
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.loss_weights = loss_weights
        self.epsilon_base = epsilon_base
        self.epsilon_plateau = epsilon_plateau
        self.epsilon_max = epsilon_max           
        
    # Training function for each GPU process
    def train(self, rank, world_size):
        
        self.model = self.model.to(rank)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=False)
        print(f"model sent to device {rank}")


        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        
        print(f"Device {rank}: loss_fn and optimizer initialized", flush=True)


    
        # Load Dataset with Distributed Sampler
        transform = transforms.Compose([transforms.ToTensor()])
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(self.train_dataset, 
                                  batch_size=self.batch_size, 
                                  sampler=train_sampler)
        
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size//4, shuffle=True)

        print(f"Device {rank}: train_loader ({len(train_loader)} samples) and val_loader ({len(val_loader)} samples) created", flush=True)
    
        stats = {
            "epoch": [],
            "step": [],
            "epsilon": [],
            "transform": [],
            "mean": [],
            "std": [],   
            "std_eps": [],
            "mse":[],
            "spec":[],
            "cos":[],
            "smooth":[],
            "hat":[],
            "lr":[]
        }

    
        losses = {
            "loss":[],
            "mse":[],
            "spec":[],
            "cos":[],
            "smooth":[],
            "hat":[],
            "sparse":[],
            "val_loss":[],
            "val_mse":[],
            "val_spec":[],
            "val_cos":[],
            "val_smooth":[],
            "val_hat":[],      
            "val_sparse":[]
        }
        
    
        mse_loss_fn = nn.MSELoss()

        for epoch in range(self.num_epochs):
             
            self.model.train() 

            loss_stats_updated = False
            
            epsilon_base = self.epsilon_base
            epsilon_plateau = self.epsilon_plateau
            epsilon_max = self.epsilon_max             

            n_plateau_start = int(self.num_epochs*len(train_loader)*0.5)
            n_plateau_end = int(self.num_epochs*len(train_loader)*0.5)         
            
            tot_loss = 0
            tot_mse_loss = 0
            tot_cosine_loss = 0
            tot_smooth_loss = 0
            tot_hat_loss = 0
            tot_spec_loss = 0
            tot_sparse_loss = 0
            
            for batch_idx, (S_new, S_in, fields, _, selected) in enumerate(train_loader):
                    
                torch.cuda.empty_cache()
                stats_updated = False
                
                # if epoch < self.num_epochs*0.1:
                #     transform_ids = [2]
                # elif epoch < self.num_epochs*0.2:
                #     transform_ids = [3]
                # elif epoch < self.num_epochs*0.3:
                #     transform_ids = [0]
                # elif epoch < self.num_epochs*0.4:
                #     transform_ids = [1]
                # else:
                #     transform_ids = [0,1,2,3]   
                #     random.shuffle(transform_ids)

                transform_ids = [0,1,2,3] # [0,1,2]   
                random.shuffle(transform_ids)                
            
                self.train_dataset.random_transform_ids = transform_ids

                # Increase ε step-wise 
                n_steps = self.num_epochs*len(train_loader)
                cur_steps = len(train_loader)*epoch + batch_idx

                if 0 <= cur_steps < n_plateau_start:
                    # print("warmup")
                    stage = 'Warmup'
                    epsilon = epsilon_base + cur_steps*(epsilon_plateau-epsilon_base)/n_plateau_start
                elif n_plateau_start <= cur_steps < n_plateau_end:
                    # print("Plateau")
                    stage = 'Plateau'
                    epsilon = epsilon_plateau
                else:
                    # print("Tune")
                    stage = 'Tuning'
                    epsilon = epsilon_plateau + (cur_steps-n_plateau_end)*(epsilon_max-epsilon_plateau)/(n_steps-n_plateau_end)
                    
                epsilon_dict = {
                    't_stretch': epsilon/2,
                    'f_stretch': epsilon,
                    'warp_2d': epsilon,
                    'amplitude': epsilon/5,
                    'phase': epsilon,
                }  
                self.train_dataset.epsilon_dict = epsilon_dict
                
                S_new = S_new.unsqueeze(1) # add channel dim
                S_new_padded = pad_spectrogram(S_new)
                S_norm = normalize_spectrogram(S_new_padded, min_val=-7.0, max_val=1.0)

                pred_fields = self.model(S_norm)
                pred_fields = pred_fields[:,:,:80,:]

                fields_norm = normalize_fields(fields,norm_ranges=epsilon_dict)
                
                true_fields = [None] * 5
                true_fields[0] = fields_norm['t_stretch']
                true_fields[1] = fields_norm['f_stretch']
                true_fields[2] = fields_norm['warp_2d'][0]
                true_fields[3] = fields_norm['warp_2d'][1]
                true_fields[4] = fields_norm['amplitude']
                # true_fields[5] = fields_norm['phase']
        
                true_fields = torch.stack(true_fields,dim=0) # C, B, F, T
                true_fields = true_fields.permute(1,0,2,3) # B, C, F, T
                
                S_recon = apply_inverse_transform(S_norm[:,:,:80,:], pred_fields, epsilon_dict=epsilon_dict)
                S_in_norm = normalize_spectrogram(S_in, min_val=-7.0, max_val=1.0)              
        
        
                mse_loss = mse_loss_fn(true_fields, pred_fields)
                cosine_loss = cosine_field_loss(pred_fields,true_fields)
                smooth_loss = spatial_smoothness_loss(pred_fields, weight=1.0)
                spec_loss = mse_loss_fn(S_recon, S_in_norm)
                
                v_adaptive = true_fields.norm(dim=1, keepdim=True).detach().clamp(min=0.0)
                hat_loss = hat_loss_fn(pred_fields,true_fields, v=v_adaptive)


                sparse_loss = sparsity_loss(
                    pred_fields, 
                    field_names=['t_stretch', 'f_stretch', 'warp2d_v', 'warp2d_w', 'amplitude'], 
                    selected_fields=['t_stretch', 'warp2d_v', 'warp2d_w'], 
                    weight=1.0
                )

                
                tot_mse_loss += mse_loss.item()
                tot_cosine_loss += cosine_loss.item()
                tot_smooth_loss += smooth_loss.item()
                tot_spec_loss += spec_loss.item()
                tot_hat_loss += hat_loss.item()
                tot_sparse_loss += sparse_loss.item()
                
                # loss = self.loss_weights["mse"]*mse_loss + self.loss_weights["cos"]*cosine_loss + self.loss_weights["spec"]*spec_loss + \
                # self.loss_weights["smooth"]*smooth_loss + self.loss_weights["phi4"]*tot_phi4_loss

                loss = self.loss_weights["cos"]*cosine_loss + self.loss_weights["spec"]*spec_loss + \
                self.loss_weights["smooth"]*smooth_loss + self.loss_weights["hat"]*hat_loss + sparse_loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                                
                if stats_updated == False:
                    stats["epoch"].append(epoch)
                    stats["step"].append(epoch*len(train_loader) + batch_idx)
                    stats["epsilon"].append(epsilon)
                    stats["transform"].append(selected[0][0])
                    stats["std"].append(pred_fields.std().item())
                    stats["mean"].append(pred_fields.mean().item())
                    stats["std_eps"].append(pred_fields.std().item()/epsilon) 
                    stats["mse"].append(mse_loss.item())
                    stats["spec"].append(spec_loss.item())
                    stats["cos"].append(cosine_loss.item())
                    stats["smooth"].append(smooth_loss.item())
                    stats["hat"].append(hat_loss.item())  
                    stats["lr"].append(optimizer.param_groups[0]['lr'])
                    stats_updated = True

                    # running_stats = {key: stats[key][-1] for key in stats.keys()}
                    # pp(running_stats)
                
                if (batch_idx)%10 == 0 and rank == 0:
                    print(f"\nStep {batch_idx}/{len(train_loader)}. Loss: {loss.item():.6f} ")
                    print(f"MSE={mse_loss.item():.6f}, cosine {cosine_loss.item():.4f}, spec={spec_loss.item():.4f}")
                    print(f"Smooth: {smooth_loss.item()}, hat: {hat_loss.item()}, sparse: {sparse_loss.item()}")
                    print(f"Fields std: {pred_fields.std().item()}, std/epsilon: {pred_fields.std().item()/epsilon:.4f}")
                    print(f"Fields abs: {pred_fields.abs().mean().item()}")
                    for k,v in fields_norm.items():
                        if k != 'warp_2d':
                            print(f"{k} field: abs = {v.abs().mean().item()}")
                        else:
                            print(f"{k} field: abs = {v[0].abs().mean().item()}, {v[1].abs().mean().item()}")
                    print(f"Epsilon={epsilon:.4f}, stage = {stage}, transformation={selected[0][0]}, cur_steps {cur_steps}")

            
                tot_loss += loss.item()
                torch.cuda.empty_cache()
        
                torch.cuda.empty_cache()
            
            ####################################################################################################
            # END TRAIN operations
            ####################################################################################################
            
            if rank == 0 and (epoch+1)%5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),  # Use `.module` to remove DDP wrapper
                }, self.model_name+"_"+str(epoch+1)+"_epochs.pth")                

            
            self.model.eval()

            val_loss = 0
            val_mse_loss = 0
            val_cosine_loss = 0
            val_smooth_loss = 0
            val_hat_loss = 0
            val_spec_loss = 0 
            val_sparse_loss = 0

            with torch.no_grad():
                for batch_idx, (S_new, S_in, fields, _, selected) in enumerate(val_loader):    
                    
                    # if epoch < self.num_epochs*0.1:
                    #     transform_ids = [2]
                    # elif epoch < self.num_epochs*0.2:
                    #     transform_ids = [3]
                    # elif epoch < self.num_epochs*0.3:
                    #     transform_ids = [0]
                    # elif epoch < self.num_epochs*0.4:
                    #     transform_ids = [1]
                    # else:
                    #     transform_ids = [0,1,2,3]   
                    #     random.shuffle(transform_ids)

                    transform_ids = [0,1,2,3] # [0,1,2]   
                    random.shuffle(transform_ids) 
                
                    self.val_dataset.random_transform_ids = transform_ids
                    
                    epsilon_dict = {
                        't_stretch': epsilon/2,
                        'f_stretch': epsilon,
                        'warp_2d': epsilon,
                        'amplitude': epsilon/5,
                        'phase': epsilon,
                    }    
                    self.val_dataset.epsilon_dict = epsilon_dict

                    S_new = S_new.unsqueeze(1) # add channel dim
                    S_new_padded = pad_spectrogram(S_new)  

                    S_norm = normalize_spectrogram(S_new_padded, min_val=-7.0, max_val=1.0)
        
                    pred_fields = self.model(S_norm)
                    pred_fields = pred_fields[:,:,:80,:]
                    
                    fields_norm = normalize_fields(fields,norm_ranges=epsilon_dict)
                    
                    true_fields = [None] * 5
                    true_fields[0] = fields_norm['t_stretch']
                    true_fields[1] = fields_norm['f_stretch']
                    true_fields[2] = fields_norm['warp_2d'][0]
                    true_fields[3] = fields_norm['warp_2d'][1]
                    true_fields[4] = fields_norm['amplitude']

                    true_fields = torch.stack(true_fields,dim=0) # C, B, F, T
                    true_fields = true_fields.permute(1,0,2,3) # B, C, F, T

                    S_recon = apply_inverse_transform(S_norm[:,:,:80,:], pred_fields, epsilon_dict=epsilon_dict)
                    S_in_norm = normalize_spectrogram(S_in, min_val=-7.0, max_val=1.0) 
                    
                    mse_loss = mse_loss_fn(true_fields, pred_fields)
                    cosine_loss = cosine_field_loss(pred_fields,true_fields)
                    smooth_loss = spatial_smoothness_loss(pred_fields, weight=1.0)
                    spec_loss = mse_loss_fn(S_recon, S_in_norm)

                    v_adaptive = true_fields.norm(dim=1, keepdim=True).detach().clamp(min=0.0)
                    hat_loss = hat_loss_fn(pred_fields,true_fields, v=v_adaptive)

    
                    sparse_loss = sparsity_loss(
                        pred_fields, 
                        field_names=['t_stretch', 'f_stretch', 'warp2d_v', 'warp2d_w', 'amplitude'], 
                        selected_fields=['t_stretch', 'warp2d_v', 'warp2d_w'], 
                        weight=1.0
                    )

                
                    val_mse_loss += mse_loss.item()
                    val_cosine_loss += cosine_loss.item()
                    val_smooth_loss += smooth_loss.item()
                    val_spec_loss += spec_loss.item()
                    val_hat_loss += hat_loss.item()
                    val_sparse_loss += sparse_loss.item()
                    
                    # loss = self.loss_weights["mse"]*mse_loss + self.loss_weights["cos"]*cosine_loss + self.loss_weights["spec"]*spec_loss + \
                    # self.loss_weights["smooth"]*smooth_loss + self.loss_weights["phi4"]*phi4_loss

                    loss = self.loss_weights["cos"]*cosine_loss + self.loss_weights["spec"]*spec_loss + \
                    self.loss_weights["smooth"]*smooth_loss + self.loss_weights["hat"]*hat_loss + sparse_loss
                    
                    val_loss += loss.item()
                    
                    torch.cuda.empty_cache()
            
            ####################################################################################################
            # END VAL LOOP
            ####################################################################################################
            # Each epoch   
            scheduler.step()

            tot_mse_loss /= len(train_loader)
            tot_cosine_loss /= len(train_loader)
            tot_smooth_loss /= len(train_loader)
            tot_hat_loss /= len(train_loader)
            tot_spec_loss /= len(train_loader)
            tot_sparse_loss /= len(train_loader)

            val_mse_loss /= len(val_loader)
            val_cosine_loss /= len(val_loader)
            val_smooth_loss /= len(val_loader)
            val_hat_loss /= len(val_loader)
            val_spec_loss /= len(val_loader)
            val_sparse_loss /= len(val_loader)
            

            if loss_stats_updated == False:
                
                losses["loss"].append(tot_loss/len(train_loader))
                losses["mse"].append(tot_mse_loss)
                losses["spec"].append(tot_spec_loss)
                losses["cos"].append(tot_cosine_loss)
                losses["smooth"].append(tot_smooth_loss)
                losses["hat"].append(tot_hat_loss)
                losses["sparse"].append(tot_sparse_loss)
                
                losses["val_loss"].append(val_loss/len(val_loader))
                losses["val_mse"].append(val_mse_loss)
                losses["val_spec"].append(val_spec_loss)
                losses["val_cos"].append(val_cosine_loss)
                losses["val_smooth"].append(val_smooth_loss)
                losses["val_hat"].append(val_hat_loss)
                losses["val_sparse"].append(val_sparse_loss)

                loss_stats_updated = True
                
            if rank == 0:
                print(f"\n\nEpoch {epoch+1}: Loss {tot_loss/len(train_loader):.4f}. Val loss {val_loss/len(val_loader):.4f}")
                print(f"Train losses.\n Field MSE: {tot_mse_loss:.5f}, cosine: {tot_cosine_loss:.5f}, smooth: {tot_smooth_loss:.5f}, hat: {tot_hat_loss:.5f}, spec MSE: {tot_spec_loss:.5f}, sparse: {tot_sparse_loss:.5f}\n\n")
        
    
                print(f"Epoch [{epoch+1}/{self.num_epochs}]")
                # running_losses = {key: round(losses[key][-1],3) for key in losses.keys()}                
                # pp(running_losses)
                print(f"LR: {optimizer.param_groups[0]['lr']}", flush=True)

            if (epoch+1)%5 == 0:
                with open(f'{self.model_name}_epoch_{epoch+1}_results.json', 'w') as fp:
                    json.dump({"stats": stats, "losses": losses}, fp)  
                
            gc.collect()
            torch.cuda.empty_cache()
    
        print("Training Complete!")
        return {"stats": stats, "losses": losses}






    