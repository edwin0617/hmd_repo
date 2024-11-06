import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def get_act(act_func_name):
    if act_func_name.lower() == "relu":
        act_func = nn.ReLU()
    elif act_func_name.lower() == 'leakyrelu':
        act_func = nn.LeakyReLU()
    elif act_func_name.lower() == "gelu":
        act_func = nn.GELU()
    elif act_func_name.lower() == 'tanh':
        act_func = nn.Tanh()
    else:
        raise ValueError("you should add activation func in 'get_act function'")
    return act_func


# input_dim, hidden_dims, output_dim, act_func_name, dropout_rate, 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, act_func_name, dropout_rate):
        super(MLP, self).__init__()

        layers = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), 
                            nn.LayerNorm(hidden_dims[0]) , 
                            get_act(act_func_name),
                            nn.Dropout(dropout_rate),
                            )
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(get_act(act_func_name))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)



# class MHA_LSTM_simpler(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
        
#         self.rnn = nn.LSTM(
#             input_size= 40,  
#             hidden_size= 60,
#             num_layers= 3,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.1
#         )
#         self.layer_norm1 = nn.LayerNorm(120)
        
#         self.selfattn_layer = nn.MultiheadAttention(
#             embed_dim=120, 
#             num_heads=4, 
#             batch_first=True)
        
#         self.frame_linear = nn.Sequential(
#             nn.Linear(120, 40),
#             nn.LayerNorm(40),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(40, 3), 
#             # nn.Sigmoid()
#         )
#         self.murmur_linear = nn.Sequential(
#             nn.Linear(1, 2),
#             )
#         self._initialize_weights()
    
    
#     def forward(self, x, pad_mask= None):        
#         x = x.permute(0, 2, 1) # [B, D, T] >> [B, T, D]
        
#         # Bi-Rnn
#         rnn_out, h_n = self.rnn(x)
#         residual = rnn_out
        
#         # Self MultiHeadAttention
#         attn_output, attn_weights = self.selfattn_layer(rnn_out, rnn_out, rnn_out, key_padding_mask= pad_mask)

#         attn_output += residual
#         attn_output = self.layer_norm1(attn_output)
        
#         seq_pred = self.frame_linear(attn_output).permute(0, 2, 1) # [B, T, C] >> [B, C, T]

#         mm_linear_input = F.sigmoid(seq_pred).mean(dim=-1)[:, -1] # [B, C, T] >> [B, C] >> [B, Murmur]
#         murmur_pred = self.murmur_linear(mm_linear_input.unsqueeze(-1).detach())
#         return seq_pred, murmur_pred 
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 init.uniform_(m.weight, -0.1, 0.1)
#                 if m.bias is not None:
#                     init.uniform_(m.bias, -0.1, 0.1)
#             elif isinstance(m, nn.LSTM):
#                 for param in m.parameters():
#                     if param.dim() >= 2:  # weight matrices
#                         init.uniform_(param, -0.1, 0.1)
#                     else:  # bias vectors
#                         init.uniform_(param, -0.1, 0.1)
#             elif isinstance(m, nn.MultiheadAttention):
#                 init.uniform_(m.in_proj_weight, -0.1, 0.1)
#                 if m.in_proj_bias is not None:
#                     init.uniform_(m.in_proj_bias, -0.1, 0.1)
#                 init.uniform_(m.out_proj.weight, -0.1, 0.1)
#                 if m.out_proj.bias is not None:
#                     init.uniform_(m.out_proj.bias, -0.1, 0.1)


class MHA_LSTM_final(nn.Module):
    def __init__(self, model_params) -> None:
        super().__init__()
        
        self.rnn = nn.LSTM(**model_params["rnn_params"])
           
        self.selfattn_layer = nn.MultiheadAttention(**model_params["attn_module"]["MHA_params"])
        self.layer_norm1 = nn.LayerNorm(model_params["attn_module"]["layernorm_dim"])
        
        self.frame_linear = MLP(**model_params["frame_linear"])
        
        
        self.murmur_linear = nn.Sequential(
            nn.Linear(model_params["murmur_linear"]["input_dim"], model_params["murmur_linear"]["output_dim"]),
            )    
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if param.dim() >= 2:  # weight matrices
                        init.uniform_(param, -0.1, 0.1)
                    else:  # bias vectors
                        init.uniform_(param, -0.1, 0.1)
            elif isinstance(m, nn.MultiheadAttention):
                init.uniform_(m.in_proj_weight, -0.1, 0.1)
                if m.in_proj_bias is not None:
                    init.uniform_(m.in_proj_bias, -0.1, 0.1)
                init.uniform_(m.out_proj.weight, -0.1, 0.1)
                if m.out_proj.bias is not None:
                    init.uniform_(m.out_proj.bias, -0.1, 0.1)
        

    def forward(self, x, pad_mask= None):        
        x = x.permute(0, 2, 1) # [B, D, T] >> [B, T, D]
        
        # Bi-Rnn
        rnn_out, h_n = self.rnn(x)
        residual = rnn_out
        
        # Self MultiHeadAttention
        attn_output, attn_weights = self.selfattn_layer(rnn_out, rnn_out, rnn_out, key_padding_mask= pad_mask)

        attn_output += residual
        attn_output = self.layer_norm1(attn_output)
        
        seq_pred = self.frame_linear(attn_output).permute(0, 2, 1) # [B, T, C] >> [B, C, T]

        mm_linear_input = F.sigmoid(seq_pred).mean(dim=-1)[:, -1] # [B, C, T] >> [B, C] >> [B, Murmur]
        murmur_pred = self.murmur_linear(mm_linear_input.unsqueeze(-1).detach())
        return seq_pred, murmur_pred 
