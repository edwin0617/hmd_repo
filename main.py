import os
import yaml 
import pytz
import datetime
# import warnings
import random
import numpy as np
import pandas as pd



import torch 

from config import get_args
from dataset import get_dataframe_dict, Stronglabeled_Dataset, Unlabeled_Dataset
from train_validate_test import merge_df_2, train_model, predict_single_model, test
from metric import*


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def main(config, device, tuning_dict=None):
    
    set_seed(config.random_seed)
    
    df_dict = get_dataframe_dict(config)
    
    train_dataset =  Stronglabeled_Dataset(config.train_datapath, 
                            df_dict["train_recording_df"].index, 
                            df_dict["train_recording_df"].murmur_timing, 
                            df_dict["train_recording_df"].patient_murmur_label, 
                            df_dict["train_recording_df"].outcome_label,
                            sampling_rate= config.sampling_rate,
                            window_length= config.window_length,
                            window_step= config.hop_length,
                            clean_noise= config.clean_noise
                            )
    unlabeled_dataset = Unlabeled_Dataset(config.external_datapath, 
                                        df_dict["unlabeled_df"].mid_path, 
                                        df_dict["unlabeled_df"].filename, 
                                        config.sampling_rate, 
                                        config.window_length, 
                                        config.hop_length, 
                                        config.freq_bins)

    val_dataset = Stronglabeled_Dataset(config.train_datapath, 
                                df_dict["val_recording_df"].index, df_dict["val_recording_df"].murmur_timing, 
                                df_dict["val_recording_df"].patient_murmur_label, df_dict["val_recording_df"].outcome_label,
                                sampling_rate= config.sampling_rate,
                                window_length= config.window_length,
                                window_step= config.hop_length,
                                clean_noise= config.clean_noise
                                )

    student, teacher = train_model(config, train_dataset, unlabeled_dataset, val_dataset, device, tuning_dict)
    
    val_fold_results_stu = predict_single_model(config, df_dict["val_recording_df"] , config.train_datapath, student, device)
    val_fold_results_tch = predict_single_model(config, df_dict["val_recording_df"] , config.train_datapath, teacher, device)   
    assert len(val_fold_results_stu) == len(val_fold_results_tch)
    
    
    # Student
    rec_predictions_df_stu = pd.DataFrame.from_dict(val_fold_results_stu, orient="index")
    recording_df_stu = df_dict["recording_df"].merge(rec_predictions_df_stu, left_index=True, right_index=True)        
    
    # Teacher
    rec_predictions_df_tch = pd.DataFrame.from_dict(val_fold_results_tch, orient="index")
    recording_df_tch = df_dict["recording_df"].merge(rec_predictions_df_tch, left_index=True, right_index=True)
    
    merged_df_stu = merge_df_2(recording_df_stu, df_dict["patient_df"])
    merged_df_tch = merge_df_2(recording_df_tch, df_dict["patient_df"])    

    
    # Get Student threshold
    optim_thr_stu= None 
    val_murmur_wma_stu= 0.0       
    
    for threshold in np.arange(0, 1.01, 0.01):
        val_murmur_preds = {} 
        
        for index, row in merged_df_stu.iterrows():
            murmur_pred = decide_murmur_with_threshold(row.to_dict(), threshold)
            val_murmur_preds[index] = {
            "prediction": murmur_pred, 
            "probabilities": [], 
            "label": row["murmur_label"]}    
        murmur_score = compute_cross_val_weighted_murmur_accuracy(val_murmur_preds, print= False)            
    
        if val_murmur_wma_stu < murmur_score:
            val_murmur_wma_stu = murmur_score.item()
            optim_thr_stu = threshold.item()      
    
    # test_patient_df, test_recording_df, data_folder, model, optim_threshold, device):
    
    test_murmur_wma_stu = test(config, df_dict["test_patient_df"], df_dict["test_recording_df"], config.test_datapath, student, optim_thr_stu, device)      
    
    
    # Teacher
    optim_thr_tch= None
    val_murmur_wma_tch = 0.0
    
    for threshold in np.arange(0, 1.01, 0.01):
        val_murmur_preds = {}  
        
        for index, row in merged_df_tch.iterrows():
            murmur_pred = decide_murmur_with_threshold(row.to_dict(), threshold)
            val_murmur_preds[index] = {
                    "prediction": murmur_pred, 
                    "probabilities": [], 
                    "label": row["murmur_label"]}
            murmur_score = compute_cross_val_weighted_murmur_accuracy(val_murmur_preds, print= False)
    
            if val_murmur_wma_tch < murmur_score:
                val_murmur_wma_tch = murmur_score.item()
                optim_thr_tch = threshold.item()       
    
    test_murmur_wma_tch = test(config, df_dict["test_patient_df"], df_dict["test_recording_df"], config.test_datapath, teacher, optim_thr_stu, device)    
    
    print(f"Test_WMA_stu: {test_murmur_wma_stu:.5f}, Test_WMA_tch: {test_murmur_wma_tch:.5f}")
    
    exp_results_dict = {"optim_threshold_stu": optim_thr_stu, 
                        "optim_threshold_tch": optim_thr_tch, 
                        "Val_WMA_stu": val_murmur_wma_stu, 
                        "Val_WMA_tch": val_murmur_wma_tch, 
                        "Test_WMA_stu": test_murmur_wma_stu, 
                        "Test_WMA_tch": test_murmur_wma_tch,
                        }
    
    return exp_results_dict



if __name__ == '__main__':
        
    config = get_args()
    
    # Select only one Gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
    GPU_NUM = 0 # Since we Selected ONE specific gpu in environment, index number is 0
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device) # Set gpu    
    
    start_time = datetime.datetime.now()
    
    if config.debug:
        config.save_path = config.save_path / "Forcheck"
    else:
        # If config.save_path is None, Set path as start timeline
        current_time = datetime.datetime.utcnow()
        kst_timezone = pytz.timezone('Asia/Seoul')
        kst_now = current_time.astimezone(kst_timezone)
        filename = kst_now.strftime("%Y-%m-%d_%H:%M:%S")
        config.save_path = config.save_path / filename
        
    os.makedirs(config.save_path, exist_ok= True)
        
    exp_result = main(config, device, tuning_dict=None)
    config = vars(config) # args >> dict
    config["exp_result"] = exp_result # Add experiment results
    
    config_result_dir = config["save_path"] / "configs_result.yaml"
    
    
    with open(config_result_dir, "w") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    
    print(f"작업 수행 시간 (초): {elapsed_time.total_seconds() // 60}분")
    print("Done.")