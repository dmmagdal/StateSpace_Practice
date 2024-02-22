
import math
from dataclasses import dataclass
from typing import Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops import rearrange, repeat
from scans import selective_scan


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    scan_mode: str = 'cumsum'


    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple
            )


class Mamba(layers.Layer):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.embedding = layers.Embedding(args.vocab_size, args.d_model)
        self.layers = [ResidualBlock(args) for _ in range(args.n_layer)]
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = layers.Dense(args.vocab_size, use_bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def call(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        return self.lm_head(x)


class ResidualBlock(layers.Layer):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)


    def call(self, x):
        return self.mixer(self.norm(x)) + x


class MambaBlock(layers.Layer):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = layers.Dense(
            args.d_inner * 2, use_bias=args.bias
        )

        self.conv1d = layers.Conv1D(
            args.d_inner,   # filter
            args.d_conv,    # kernel
            use_bias=args.conv_bias,
            groups=args.d_inner,
            padding="causal" if args.d_conv - 1 else "same"
        )
        
        self.x_proj = layers.Dense(
            args.dt_rank + args.d_state * 2, use_bias=False
        )

        self.dt_proj = layers.Dense(
            args.dt_rank, args.d_inner, use_bias=True
        )

        A = repeat(tf.range(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = tf.Variable(tf.math.log(A))
        self.D = tf.Variable(tf.ones(args.d_inner))
        self.out_proj = tf.Variable(args.d_inner, args.d_model, use_bias=args.bias)


    def call(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = tf.split(
            x_and_res, 
            num_or_size_splits=[self.args.d_inner, self.args.d_inner], 
            axis=-1
        )

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = tf.keras.activations.silu(x)

        y = self.ssm(x)
        
        y = y * tf.keras.activations.silu(res)
        
        return self.out_proj(y)


    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -tf.math.exp(tf.cast(self.A_log, dtype=tf.float32))  # shape (d_in, n)
        D = tf.cast(self.D, dtype=tf.float32)

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = tf.split(
            x_dbl, 
            num_or_size_splits=[self.args.dt_rank, n, n], 
            axis=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = tf.keras.activations.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        return selective_scan(
            x, delta, A, B, C, D, mode=self.args.scan_mode
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]


class RMSNorm(layers.Layer):
    def __init__(self, d_model: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = tf.Variable(tf.ones(d_model))


    def call(self, x):
        output = tf.math.rsqrt(
            tf.math.reduce_mean(
                tf.math.pow(x, 2),
                axis=-1, 
                keepdims=True
            ) + self.eps
        ) * self.weight
        return output