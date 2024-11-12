"""
-----------------------------------------------------------------------------
Transformer using pytorch and numpy
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
Package required:
Numpy: https://numpy.org/
Matplotlib: https://matplotlib.org
-----------------------------------------------------------------------------
"""

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import math
from typing import Tuple


class self_attention_layer(nn.Module):
    """
    Self attention layer
    """
    
    def __init__(
        self,
        dims_embd: int,
    )->None:
        """
        Self attention class initialization
        
        Inpout:
            - dims_embd (int): Embedding dimension
        """
        
        super().__init__()
        self.dims_embd_ = dims_embd
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        self.W_q_ = nn.Linear(dims_embd, dims_embd)
        self.W_k_ = nn.Linear(dims_embd, dims_embd)
        self.W_v_ = nn.Linear(dims_embd, dims_embd)


        
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self, 
        x: Tensor 
    )->Tensor:
        """
        Forward pass for the self attention layer
        
        Imput:
            - x (torch tensor): Input data
            
        Output:
        
        """
        
        y = []
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        Q=self.W_q_(x)
        K=self.W_k_(x)
        V=self.W_v_(x)

        temp = torch.bmm(Q, K.transpose(1, 2))
        temp=temp/(self.dims_embd_) ** (1/2.)
        temp = F.softmax(temp, dim=2)

        y = torch.bmm(temp, V)
        



        

        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
    
        return y
    
class transformer_block_encoder(nn.Module):
    """
    Transformer single block
    """
    
    def __init__(
        self,
        dims_embd: int,
        num_hidden_nodes_ffnn: int = 2048,
        dropout_prob: float = 0.0
    )->None:
        """
        Transformer single block class initialization
        
        Inpout:
            - dims_embd (int):             Embedding dimension
            - num_hidden_nodes_ffnn (int): Number of neurons in the fed-forward layer
            - dropout_prob (float):        Dropout probability in liner layers
        """
        
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        self.attention_ = self_attention_layer(dims_embd)
        
        self.layer_norm1_ = nn.LayerNorm(dims_embd)
        self.layer_norm2_ = nn.LayerNorm(dims_embd)
        
        self.ffnn_ = nn.Sequential(
            nn.Linear(dims_embd, num_hidden_nodes_ffnn),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes_ffnn, dims_embd)
        )
        self.droput_ops_ = nn.Dropout(dropout_prob)
        
        self.dims_embd_ = dims_embd
        self.num_hidden_nodes_ffnn_ = num_hidden_nodes_ffnn
        self.dropout_prob_ = dropout_prob
        
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self,
        x: Tensor,
    )->Tensor:
        """
        Forward pass for the transformer block
        
        Imput:
            - x (torch tensor): Input data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        x=self.layer_norm1_(x + self.attention_(x))
        x=self.layer_norm2_(x + self.ffnn_(x))
    
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
        return x
        
class transformer_encoder(nn.Module):
    """
    Transformer encoder module
    """
    
    def __init__(
        self,
        dims_embd: int,
        num_hidden_nodes_ffnn: int = 2048,
        dropout_prob: float = 0.0,
        num_layers_encoder: int = 2
    )->None:
        """
        Transformer encoder class initialization
        
        Inpout:
            - dims_embd (int):             Embedding dimension
            - num_hidden_nodes_ffnn (int): Number of neurons in the fed-forward layer
            - dropout_prob (float):        Dropout probability in liner layers
            - num_layers_encoder (int):    Number encoder blocks
        """
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        self.trs_endr_blocks_ = nn.ModuleList(
            [
                transformer_block_encoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob) for _ in range(num_layers_encoder)
            ]
        )
        
        self.num_layers_encoder_ = num_layers_encoder
        
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
    
    def forward(
        self,
        x: Tensor,
    )->Tensor:
        """
        Forward pass for the transformer encoder
        
        Imput:
            - x (torch tensor): Input data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        for i in range(self.num_layers_encoder_):
            x = self.trs_endr_blocks_[i](x)
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
        return x
    
    
class cross_attention_layer(nn.Module):
    """
    Cross attention layer
    """
    
    def __init__(
        self,
        dims_embd: int,
    )->None:
        """
        Cross attention class initialization
        
        Inpout:
            - dims_embd (int): Embedding dimension
        """
        
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        self.W_q_ = nn.Linear(dims_embd, dims_embd)
        self.W_k_ = nn.Linear(dims_embd, dims_embd)
        self.W_v_ = nn.Linear(dims_embd, dims_embd)

        # self.mask=  torch.tril(torch.ones())
        
        self.dims_embd_ = dims_embd

        

        
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self, 
        x: Tensor,
        y: Tensor
    )->Tensor:
        """
        Forward pass for the cross-attention layer
        
        Imput:
            - x (torch tensor): Input encoder data
            - y (torch tensor): Input decoder data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        Q = self.W_q_(x)
        K = self.W_k_(y)
        V = self.W_v_(y)
        
        # Scaled dot-product attention
        temp = torch.bmm(Q, K.transpose(1, 2))
        temp = temp / (self.dims_embd_ ** 0.5)
        temp = F.softmax(temp, dim=-1)

        y = torch.bmm(temp, V)


        
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
    
        return y
    

class transformer_block_decoder(nn.Module):
    """
    Transformer single decoder block
    """
    
    def __init__(
        self,
        dims_embd: int,
        num_hidden_nodes_ffnn: int = 2048,
        dropout_prob: float = 0.0
    )->None:
        """
        Transformer single block class initialization
        
        Inpout:
            - dims_embd (int):             Embedding dimension
            - num_hidden_nodes_ffnn (int): Number of neurons in the fed-forward layer
            - dropout_prob (float):        Dropout probability in liner layers
        """
        
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        self.attention_ = self_attention_layer(dims_embd)
        self.cross_attention_ = cross_attention_layer(dims_embd)
        
        self.layer_norm1_ = nn.LayerNorm(dims_embd)
        self.layer_norm2_ = nn.LayerNorm(dims_embd)
        self.layer_norm3_ = nn.LayerNorm(dims_embd)
        
        self.ffnn_ = nn.Sequential(
            nn.Linear(dims_embd, num_hidden_nodes_ffnn),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes_ffnn, dims_embd)
        )
        self.droput_ops_ = nn.Dropout(dropout_prob)
        
        self.dims_embd_ = dims_embd
        self.num_hidden_nodes_ffnn_ = num_hidden_nodes_ffnn
        self.dropout_prob_ = dropout_prob
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self,
        x: Tensor,
        y: Tensor
    )->Tensor:
        """
        Forward pass for the transformer block
        
        Imput:
            - x (torch tensor): Input encoder data
            - y (torch tensor): Input decoder data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        x = self.layer_norm1_(x + self.attention_(x))
        x = self.droput_ops_(x)
        x = self.layer_norm2_(x + self.cross_attention_(x, y))
        x = self.droput_ops_(x)
        x = self.layer_norm3_(x + self.ffnn_(x))
        x = self.droput_ops_(x)
    
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
        return x
    
    
class transformer_decoder(nn.Module):
    """
    Transformer decoder module
    """
    
    def __init__(
        self,
        dims_embd: int,
        num_hidden_nodes_ffnn: int = 2048,
        dropout_prob: float = 0.0,
        num_layers_decoder: int = 2
    )->None:
        """
        Transformer decoder class initialization
        
        Inpout:
            - dims_embd (int):             Embedding dimension
            - num_hidden_nodes_ffnn (int): Number of neurons in the fed-forward layer
            - dropout_prob (float):        Dropout probability in liner layers
            - num_layers_decoder (int):    Number decoder blocks
        """
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        self.trs_dcdr_blocks_ = nn.ModuleList(
            [
                transformer_block_decoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob) for _ in range(num_layers_decoder)
            ]
        )
        
        self.num_layers_decoder_ = num_layers_decoder
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self,
        x: Tensor,
        y: Tensor
    )->Tensor:
        """
        Forward pass for the transformer encoder
        
        Imput:
            - x (torch tensor): Input encoder data
            - y (torch tensor): Input decoder data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        for block in self.trs_dcdr_blocks_:
            x = block(x, y)
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
        return x
        
       



class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()     

    # initialize dropout                  
    self.dropout = nn.Dropout(p=dropout)      

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)    

    # create position column   
    k = torch.arange(0, max_length).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)  

    # add dimension     
    pe = pe.unsqueeze(0)          

    # buffers are saved in state_dict but not trained by the optimizer                        
    self.register_buffer("pe", pe)                        

  def forward(self, x: Tensor):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.size(1)].requires_grad_(False) 

    # perform dropout
    return self.dropout(x)

    




dims_embd = 10
num_data_points = 100
batch_size = 5
num_hidden_nodes_ffnn = 1024
dropout_prob = 0.2
num_layers_encoder = 2

x = torch.rand(batch_size, num_data_points, dims_embd)
y = torch.rand(batch_size, num_data_points, dims_embd)

emb = nn.Linear(dims_embd, dims_embd)
x=emb(x)
y=emb(y)


p=PositionalEncoding(dims_embd,dropout_prob,num_data_points)

x=p(x)


y=p(y)





# Test Self-attention layer and its input output size  
print('='*70)
model_self_attention_layer = self_attention_layer(dims_embd)
print('Self-attention layer models is: \n{}' .format(model_self_attention_layer))
print('-'*70)

y_bar = model_self_attention_layer(x)
print('Self-attention layer input size: {}' .format(x.shape))
print('Self-attention layer output size: {}' .format(y_bar.shape))
print('-'*70)
        
# Test Transformer encoder block input output size 
print('='*70)
model_transformer_block_encoder = transformer_block_encoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob)
print('Transformer block models is: \n{}' .format(model_transformer_block_encoder))
print('-'*70)
print('Transformer block models summary:')
print('-'*70)
summary(model_transformer_block_encoder, (num_data_points, dims_embd, ), device=str("cpu"))
print('-'*70)

y_bar = model_transformer_block_encoder(x)
print('Transformer block input size: {}' .format(x.shape))
print('Transformer block output size: {}' .format(y_bar.shape))  
print('-'*70)

# Test Transformer encoder input output size 
print('='*70)
model_transformer_encoder = transformer_encoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob, num_layers_encoder)
print('Transformer encoder models is: \n{}' .format(model_transformer_encoder))
print('-'*70)
print('Transformer encoder models summary:')
print('-'*70)
summary(model_transformer_encoder, (num_data_points, dims_embd, ), device=str("cpu"))
print('-'*70)

y_bar = model_transformer_encoder(x)
print('Transformer encoder input size: {}' .format(x.shape))
print('Transformer encoder output size: {}' .format(y_bar.shape))  
print('-'*70)

# # Test Cross-attention layer and its input output size  
print('='*70)
model_cross_attention_layer = cross_attention_layer(dims_embd)
print('Cross-attention layer models is: \n{}' .format(model_cross_attention_layer))
print('-'*70)

y_bar = model_cross_attention_layer(x, y)
print('Cross-attention layer input size: {}' .format(x.shape))
print('Cross-attention layer output size: {}' .format(y_bar.shape))
print('-'*70)

# Test Transformer decoder block input output size 
print('='*70)
model_transformer_block_decoder = transformer_block_decoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob)
print('Transformer decoder block models is: \n{}' .format(model_transformer_block_decoder))
print('-'*70)
print('Transformer decoder block models summary:')
print('-'*70)
summary(model_transformer_block_decoder, [(num_data_points, dims_embd, ), (num_data_points, dims_embd, )], device=str("cpu"))
print('-'*70)

y_bar = model_transformer_block_decoder(x, y)
print('Transformer block input size: {}' .format(x.shape))
print('Transformer block output size: {}' .format(y_bar.shape))  
print('-'*70)

# Test Transformer decoder input output size 
print('='*70)
model_transformer_decoder = transformer_decoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob, num_layers_encoder)
print('Transformer decoder models is: \n{}' .format(model_transformer_decoder))
print('-'*70)
print('Transformer decoder models summary:')
print('-'*70)
summary(model_transformer_decoder, [(num_data_points, dims_embd, ), (num_data_points, dims_embd, )], device=str("cpu"))
print('-'*70)

y_bar = model_transformer_decoder(x, y)
print('Transformer decoder input size: {}' .format(x.shape))
print('Transformer decoder output size: {}' .format(y_bar.shape))  
print('-'*70)
        
        
        
        