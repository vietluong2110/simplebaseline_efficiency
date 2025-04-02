from layers.JAX_Transformer_EncDec import JAX_Encoder, JAX_EncoderLayer
from layers.JAX_SelfAttention_Family import JAX_GeomAttentionLayer, JAX_GeomAttention
from layers.JAX_Embed import JAX_DataEmbedding_inverted

import pdb
import jax.numpy as jnp
from flax import nnx
import jax
class Model(nnx.Module):
    def __init__(self, configs):
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.geomattn_dropout = configs.geomattn_dropout
        self.alpha = configs.alpha
        self.kernel_size = configs.kernel_size
        # Embedding
        enc_embedding = JAX_DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.rngs, configs.dropout)
        self.enc_embedding = enc_embedding

        encoder = JAX_Encoder(
            [  # Wrap the EncoderLayer in a list
                JAX_EncoderLayer(
                    JAX_GeomAttentionLayer(
                        JAX_GeomAttention(
                            configs.rngs, 
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention, alpha=self.alpha
                        ),
                        configs.d_model, 
                        configs.rngs, 
                        requires_grad=configs.requires_grad, 
                        wv=configs.wv, 
                        m=configs.m, 
                        d_channel=configs.dec_in, 
                        kernel_size=self.kernel_size, 
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    configs.d_model,
                    configs.rngs, 
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers) # the tuned results before Nov 15th only use fixed 1 layer
            ],
            norm_layer=nnx.LayerNorm(configs.d_model, rngs = configs.rngs)
        )
        self.encoder = encoder

        projector = nnx.Linear(configs.d_model, self.pred_len, precision = jax.lax.Precision('highest'), rngs = configs.rngs)
        self.projector = projector


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(axis = 1, keepdims=True)
            x_enc = x_enc - means
            stdev = jnp.sqrt(jnp.var(x_enc, axis=1, keepdims=True) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        enc_embedding = self.enc_embedding
        encoder = self.encoder
        projector = self.projector
        # pdb.set_trace()
        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens


        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = jnp.permute_dims(projector(enc_out), (0, 2, 1))[:, :, :N] # filter the covariates


        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * jnp.tile(
                jnp.expand_dims(stdev[:, 0, :], axis=1), 
                (1, self.pred_len, 1))
            dec_out = dec_out + jnp.tile(
                jnp.expand_dims(means[:, 0, :], axis=1), 
                (1, self.pred_len, 1)
            )
        return dec_out, attns


    def __call__(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # pdb.set_trace()
        dec_out, attns = self.forecast(x_enc, None, None, None)
        return dec_out, attns  # [B, L, D]  