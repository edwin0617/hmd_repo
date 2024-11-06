import sys
import pathlib
import argparse


# TODO: set model parameters
def get_model_params():
    model_params = {}
    model_params['rnn_params'] =  {"input_size": 40, 
                                "hidden_size": 60, 
                                "num_layers": 3, 
                                "batch_first": True, 
                                "bidirectional": True, 
                                "dropout": 0.1,
                                             }
    model_params["attn_module"] = {"MHA_params": {"embed_dim": model_params['rnn_params']["hidden_size"] * 2, 
                                                    "num_heads": 4, 
                                                    "batch_first": True, },
                               "layernorm_dim": model_params['rnn_params']["hidden_size"] * 2}
    model_params["frame_linear"] = {"input_dim": model_params['rnn_params']["hidden_size"] * 2, 
                                "hidden_dims": [40],
                                "output_dim": 3,
                                "act_func_name": 'gelu', 
                                "dropout_rate": 0.3,
                                }
    model_params["murmur_linear"] = {"input_dim": 1, 
                                 "output_dim": 2}
    
    return model_params



def get_args():
    # Jupyter Notebook에서 불필요한 인자 제거
    sys.argv = [''] # To check parse_args return in ipynb file.
    
    parser = argparse.ArgumentParser(description="Load training configuration")
    
    # Virtual Machine settings
    parser.add_argument("--debug", type=bool, default=False, help="Debugging Mode to check pipline is ok")
    parser.add_argument("--gpu_index", type=str, default= "0", help="Index of 1 gpu, you can change this if you have at least 2 gpus")
    parser.add_argument("--num_workers", type=int, default= 8, help="number of workers")
    parser.add_argument("--verbose", type=str, default=1)
    
    # Path
    parser.add_argument("--save_path", type=str, default=pathlib.Path("./exps"), help="Save Path where to save parameters & results")
    parser.add_argument("--train_datapath", type=str, default=pathlib.Path("/Data2/murmur/train"))
    parser.add_argument("--test_datapath", type=str, default=pathlib.Path("/Data2/murmur/test"))
    parser.add_argument("--external_datapath", type=str, default=pathlib.Path("/Data2/heart_sound_dataset"))
    parser.add_argument("--external_data_subpath", type=str, default={"pysionet_sufhsdb": "pysionet_sufhsdb", 
                                                                      "kaggle_set_a": "kag_dataset_1/set_a", 
                                                                      "kaggle_set_b": "kag_dataset_1/set_b"})

    # Define validation fold data
    parser.add_argument("--num_k", type=int, default=5, help="number of fold")
    parser.add_argument("--val_fold_num", type=int, default=3, help="i'th val fold data")
    
    # Data preprocessing parameters
    parser.add_argument("--sampling_rate", type=int, default=4000)
    parser.add_argument("--window_length", type=float, default=0.050)
    parser.add_argument("--hop_length", type=float, default=0.020)
    parser.add_argument("--freq_high", type=int, default=800, help="to remove no heart signal part")
    parser.add_argument("--freq_bins", type=int, default=40)
    parser.add_argument("--train_seq_len", type= int, default= 6)
    parser.add_argument("--clean_noise", type=bool, default= True)

    # Model parameters to Train
    parser.add_argument("--model_params_dict", type= dict, default= get_model_params(), help= "Model parameters")
    
    # Model hyper-parameters    
    parser.add_argument("--random_seed", type=int, default=0, help="seed_number for reproduction")
    parser.add_argument("--max_epoch", type=int, default= 250)
    parser.add_argument("--train_bs", type=int, default=80, help="train batch size, Strong labeled data")
    parser.add_argument("--unlabel_bs", type=int, default=160, help="unlabeld data batch size")
    parser.add_argument("--val_bs", type=int, default=120, help="val batch size")
    parser.add_argument("--learning_rate", type=float, default= 1e-4 * 5, help="")
    parser.add_argument("--base_lr", type=float, default=4e-5, help="initial learning rate for warm-up phase")
    parser.add_argument("--max_lr", type=float, default=1e-3, help="max learning rate")
    parser.add_argument("--final_lr", type= float, default= 1e-5, help= "final learning rate")
    parser.add_argument("--training_patience", type=int, default=10, help="early-stopping condition")
    parser.add_argument("--pos_weight", type=float, default=4.0, help="positive class weight in cross-entropy loss")
    
    # Mean-Teacher(Ema) parameters
    parser.add_argument("--ema_factor", type=float, default=0.99)
    parser.add_argument("--const_max", type=float, default= 1)
    
    # Data Augmentation parameters: [Mix-up]
    parser.add_argument("--use_mixup", type=bool, default= False)
    parser.add_argument("--mixup_alpha", type=float, default= 0.2)
    parser.add_argument("--mixup_beta", type=float, default= 0.2)
    parser.add_argument("--mixup_label_type", type=str, default= "soft")
    
    # parser.add_argument("--", type=, default="", help="")
    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()

    return args

if __name__ == '__main__':
    args= get_args()
    print(args)
    