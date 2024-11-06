import random
import numpy as np
import pathlib
import math
from tqdm import tqdm
from copy import deepcopy
from itertools import cycle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import scipy.io as spio
from scipy.signal import resample

import warnings 
warnings.filterwarnings("ignore", message="stft with return_complex=False is deprecated")


# from scheduler import create_custom_lambda_scheduler
from dataset import MAX_LENGTH, FREQ_BINS, merge_df_2, calculate_features
from data_augment import mixup_strong
from model import MHA_LSTM_final
from metric import*

# from config import parse_args
# config = parse_args()






def train_model(config, train_dataset, unlabeled_dataset, val_dataset, device, tuning_dict=None):

    sampler_unlabeled = RandomSampler(unlabeled_dataset, replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size= config.train_bs, shuffle=True, collate_fn= collate_fn, num_workers=config.num_workers)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size= config.unlabel_bs, sampler= sampler_unlabeled, collate_fn= collate_fn, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size= config.val_bs, shuffle=False,  collate_fn= collate_fn, num_workers=config.num_workers)     
    
    student = MHA_LSTM_final(config.model_params_dict).to(device)
    teacher = deepcopy(student)
    student, teacher = map(lambda x: x.to(device), [student, teacher])    
    
    # Epoch
    if tuning_dict:
        training_max_epoch = tuning_dict["max_epoch"]
    else:
        training_max_epoch = 1 if config.debug else config.max_epoch
        
    # Optimizer
    optim = torch.optim.AdamW(student.parameters(), 
                              lr= tuning_dict["learning_rate"] if tuning_dict else config.learning_rate,
                            weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)
    
    
    best_total_loss, best_seq_loss, best_murmur_loss= tuple([np.inf] * 3)
    consis_weight = config.const_max
    early_stop_counter = 0
    optim_count= 0

    class_weights = torch.tensor([1.0, config.pos_weight]).float().to(device)
    crit_murmur = nn.CrossEntropyLoss(weight = class_weights)    
    
    
    for epoch in tqdm(range(1, training_max_epoch + 1)):
        student.train()
        current_lr = optim.param_groups[0]['lr']
        consis_weight *= 0.99    
        
        total_losses, strong_seq_losses, train_mm_losses, cons_seq_losses = tuple([0] * 4)
        
        for i, (batch_strong, batch_unlabel) in enumerate(zip(train_dataloader, cycle(unlabeled_dataloader))):
            
            str_mels, str_seq_labels, str_pad_masks, str_mm_labels, _, _ = batch_strong
            ul_mels, ul_seq_labels, ul_pad_masks, ul_mm_labels, _, _ = batch_unlabel
            
            batch_mels = torch.concat([str_mels, ul_mels], dim= 0).to(device)
            batch_seq_labels = torch.concat([str_seq_labels, ul_seq_labels], dim= 0).to(device)
            batch_pad_masks = torch.concat([str_pad_masks, ul_pad_masks], dim= 0).to(device)
            batch_mm_labels = torch.concat([str_mm_labels, ul_mm_labels], dim= 0).to(device)
            
            batch_num = batch_mels.shape[0]
            strong_mask = torch.zeros(batch_num).to(batch_mels).bool()
            unlabeled_mask = torch.zeros(batch_num).to(batch_mels).bool()            
            
            strong_mask[ : str_mels.shape[0]] = 1
            unlabeled_mask[str_mels.shape[0] : ] = 1            
        
            if config.use_mixup:
                if random.random() > 0.5 :
                    batch_mels[strong_mask], batch_seq_labels[strong_mask], batch_mm_labels[strong_mask]\
                        = mixup_strong(batch_mels[strong_mask], batch_seq_labels[strong_mask], batch_mm_labels[strong_mask])    
    

            optim.zero_grad()
             
            # Student
            seq_pred_stu, mm_pred_stu = student(batch_mels, batch_pad_masks)

            # Teacher
            with torch.no_grad():
                seq_pred_tch, mm_pred_tch = teacher(batch_mels, batch_pad_masks)
                
            train_seq_loss = F.binary_cross_entropy_with_logits(seq_pred_stu[strong_mask], batch_seq_labels[strong_mask], reduction= 'none')
            strong_pad_mask = batch_seq_labels[strong_mask] != -1
            strong_seq_loss = train_seq_loss[strong_pad_mask].mean()
            
            train_mm_loss = crit_murmur(mm_pred_stu[strong_mask], batch_mm_labels[strong_mask])
            
            # consistency loss
            cons_seq_loss = F.mse_loss(F.sigmoid(seq_pred_stu[unlabeled_mask]), F.sigmoid(seq_pred_tch[unlabeled_mask].detach()))
            cons_mm_loss = F.mse_loss(F.sigmoid(mm_pred_stu[unlabeled_mask]), F.sigmoid(mm_pred_tch[unlabeled_mask].detach()))   
            cons_loss = cons_seq_loss + cons_mm_loss       
            
            total_loss = (strong_seq_loss + train_mm_loss) + cons_loss * consis_weight
            
            total_losses += total_loss.cpu().item()
            strong_seq_losses += strong_seq_loss.cpu().item()
            train_mm_losses += train_mm_loss.cpu().item()
            cons_seq_losses += cons_loss.cpu().item()            
            
            # Update Student model parameters
            total_loss.backward()
            optim.step()
            optim_count += 1
            
            # Update EMA
            student, teacher = update_ema(config.ema_factor, optim_count, student, teacher)         
            
            
        # 1 epoch is done.
        with torch.no_grad():
            student.eval()
            val_total_losses, val_seq_losses, val_mm_losses = tuple([0] * 3)
            
            for i, (batch_val) in enumerate(val_dataloader):
                val_mels, val_seq_labels, val_pad_masks, val_mm_labels, _, _ = batch_val
                val_mels, val_seq_labels, val_pad_masks, val_mm_labels =\
                    map(lambda x: x.to(device), [val_mels, val_seq_labels, val_pad_masks, val_mm_labels])
                    
                val_seq_preds, val_murmur_preds = student(val_mels, val_pad_masks)
                
                val_seq_loss = F.binary_cross_entropy_with_logits(val_seq_preds, val_seq_labels, reduction= 'none')
                val_pad_mask = val_seq_labels != -1
                val_seq_loss = val_seq_loss[val_pad_mask].mean()   
    
                val_mm_loss = crit_murmur(val_murmur_preds, val_mm_labels)
                val_total_loss = val_seq_loss + val_mm_loss
    
                val_total_losses += val_total_loss.cpu().item()
                val_seq_losses += val_seq_loss.cpu().item()
                val_mm_losses += val_mm_loss.cpu().item()                
                    
            if val_mm_losses < best_murmur_loss:
                save_student_teacher_both(config.save_path, student, teacher)
                best_murmur_loss = val_mm_losses
                early_stop_counter = 0                 
                
            else:
                early_stop_counter += 1
                if early_stop_counter > config.training_patience:
                    break
                
    # Training is Done.
    model_name = type(student).__name__
    student, teacher = load_student_teacher_both(config, model_name)
    return student, teacher



def save_student_teacher_both(save_path: pathlib.Path, student: nn.Module, teacher: nn.Module):
    student_path = save_path / "Best_student.pt"
    teacher_path = save_path / "Best_teacher.pt"
    torch.save(student.state_dict(), student_path)
    torch.save(teacher.state_dict(), teacher_path)

    
def load_student_teacher_both(config, model_name: str):
    student, teacher = eval(model_name)(config.model_params_dict), eval(model_name)(config.model_params_dict)
    student.load_state_dict(torch.load(pathlib.Path(config.save_path) / "Best_student.pt"))
    teacher.load_state_dict(torch.load(pathlib.Path(config.save_path) / "Best_teacher.pt"))
    return student, teacher


def update_ema(alpha, global_step, model, ema_model): # student, teacher
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)
    return model, ema_model


def predict_single_model(config, eval_recording_df, data_folder, model, device):
    
    # eval_recording_df = Df_dict[""]
    sampling_rate = config.sampling_rate
    
    model = model.to(device)
    result = {}
    
    for filename in eval_recording_df.index:
                
        filepath = data_folder / filename
        sr, recording = spio.wavfile.read(filepath.with_suffix(".wav"))
        
        if sr != sampling_rate:    
            num_samples = int(len(recording) * sampling_rate / sr)
            recording = resample(recording, num_samples)
        recording = torch.as_tensor(recording.copy())
        
        mel, _ = calculate_features(recording, sampling_rate)
        mel = mel.unsqueeze(0)
        mel = mel.to(device)
        
        # _, murmur_pred = model(mel, actual_length)
        _, murmur_pred = model(mel)
        murmur_pred = F.softmax(murmur_pred, dim= -1) # Added
        
        result[filename] = {"holo_HSMM": murmur_pred.squeeze(0)[1].detach().cpu().item()}
        
    return result


def test(config, test_patient_df, test_recording_df, test_data_folder, model, optim_threshold, device):
    
    test_results = predict_single_model(config, test_recording_df, test_data_folder, model, device)
    
    
    test_predictions_df = pd.DataFrame.from_dict(test_results, orient="index")
    test_recording_df = test_recording_df.merge(test_predictions_df, left_index=True, right_index=True)
    
    merged_df = merge_df_2(test_recording_df, test_patient_df)
    
    test_murmur_preds= {}
    
    for index, row in merged_df.iterrows():
        murmur_pred =  decide_murmur_with_threshold(row.to_dict(), optim_threshold)
        test_murmur_preds[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}
    
    test_murmur_score = compute_cross_val_weighted_murmur_accuracy(test_murmur_preds, print= False)
    
    return test_murmur_score.item()        






def collate_fn(x):
        
    all_features = torch.zeros(len(x), FREQ_BINS, MAX_LENGTH)  # [bs, freq, frame]
    all_seq_labels = torch.ones(len(x), 3, MAX_LENGTH) * (-1) # S1, S2, Murmur, 
    all_murmur_labels = []
    all_outcome_labels = []
    pad_masks = []    
    all_filenames = []
    
    for idx, (features, seq_label, murmur_label, outcome_label, wav_len, filename) in enumerate(x):
        
        pad_mask = torch.ones(MAX_LENGTH)
        
        # 같거나 짧음
        if features.shape[-1] <= MAX_LENGTH:
            diff = MAX_LENGTH - features.shape[-1]
            start = random.randint(0, diff)
            end = start + features.shape[-1]
            all_features[idx, :, start : end] = features
            all_seq_labels[idx, : , start : end] = seq_label
            
            pad_mask[start:end] = 0.0 # 데이터 있는 부분이 0.0
            pad_masks.append(pad_mask)
            # actual_seq_lengths.append(len(seq_label))
            all_filenames.append(filename)
        # 더 길면
        else:
            diff = features.shape[-1] - MAX_LENGTH
            start = random.randint(0, diff)
            end =  start + MAX_LENGTH
            all_features[idx, :, :] = features[:, start : end]
            all_seq_labels[idx, :, :] = seq_label[:, start : end]
            
            pad_mask[:] = 0.0 # 모든 시퀀스에 데이터 있으므로 모두 0.0
            pad_masks.append(pad_mask)
            # actual_seq_lengths.append(max_length)

        all_murmur_labels.append(murmur_label)
        all_outcome_labels.append(outcome_label)
        all_filenames.append(filename)
        
    all_features = all_features.float()
    all_seq_labels = all_seq_labels.float()
    all_murmur_labels = torch.stack(all_murmur_labels).float()
    pad_masks = torch.stack(pad_masks).bool() #.float()

    return (all_features, all_seq_labels, pad_masks, all_murmur_labels, all_outcome_labels, all_filenames)


