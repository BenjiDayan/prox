import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        # @Benji: guessing that token_embedding's 0th dim is token number, and this is < max_len, presumably mostly equal.
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(1), :].squeeze(1))

class PoseTransformer(nn.Module):
    def __init__(
        self,
        num_tokens=75,
        dim_model=200,
        max_seq_len=30,
        num_heads=2,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout_p=0.1,
    ):
        super().__init__()
        self.dim_model = dim_model

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=max_seq_len
        )

        # This is our embedding function (just as NLP would use word2vec to embed each word as a vector).
        self.fc1 = nn.Linear(num_tokens, dim_model//2)
        self.fc2 = nn.Linear(dim_model//2, dim_model)

        # Layers
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )

        self.out1 = nn.Linear(dim_model, dim_model//2)
        self.out2 = nn.Linear(dim_model//2, num_tokens)

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def forward_predict(self, src, pred_frames: int, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        src_enc = self.fc2(self.fc1(src)) * math.sqrt(self.dim_model)
        src_enc = self.positional_encoder(src_enc)
        src_enc = src_enc.permute(1, 0, 2)

        last_out = None
        outputs = []
        for i in range(pred_frames):
            if i == 0:
                tgt = src[:, -1, :].unsqueeze(1)  # (batch_size, 1, dim_model)
            else:
                tgt = torch.cat([tgt, last_out], dim=1)

            tgt_mask = self.get_tgt_mask(i+1).to(device)
            tgt_enc = self.fc2(self.fc1(tgt)) * math.sqrt(self.dim_model)
            tgt_enc = tgt_enc.permute(1, 0, 2)

            # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
            transformer_out = self.transformer(src_enc, tgt_enc, tgt_mask=tgt_mask, src_key_padding_mask=None,
                                               tgt_key_padding_mask=None)
            out = self.out2(self.out1(transformer_out))

            last_out = out.permute(1, 0, 2)[:, -1:, :]
            outputs.append(last_out)

        return torch.cat(outputs, dim=1)


    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src_enc = self.fc2(self.fc1(src)) * math.sqrt(self.dim_model)
        tgt_enc = self.fc2(self.fc1(tgt)) * math.sqrt(self.dim_model)
        src_enc = self.positional_encoder(src_enc)
        tgt_enc = self.positional_encoder(tgt_enc)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src_enc = src_enc.permute(1,0,2)
        tgt_enc = tgt_enc.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src_enc, tgt_enc, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out2(self.out1(transformer_out))

        out = out.permute(1,0,2)
        # TODO is correct?
        # out += src[:,-1,:].unsqueeze(1)
        return out
