import tensorflow as tf
import enum
import math
class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')
    
    def apply(self, n_features: int, d_token: int) -> tf.Variable:
        d_sqrt_inv = 1 / math.sqrt(d_token)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            initializer = tf.random_uniform_initializer(minval=-d_sqrt_inv, maxval=d_sqrt_inv)
            return tf.Variable(initial_value=initializer(shape=(n_features, d_token), dtype="float32"),trainable=True,)
            #nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            #nn.init.normal_(x, std=d_sqrt_inv)
            initializer = tf.random_normal_initializer(stddev=d_sqrt_inv)
            return tf.Variable(initial_value=initializer(shape=(n_features, d_token), dtype="float32"),trainable=True,)

class NumericalFeatureTokenizer(tf.keras.layers.Layer):
    def __init__(
            self,
            n_features: int,
            d_token: int,
            bias: bool,
            initialization: str,
            ) -> None:
        super(NumericalFeatureTokenizer, self).__init__()
        
        initialization_ = _TokenInitialization.from_str(initialization)
        
        self.weight = initialization_.apply(n_features, d_token)
        self.bias = initialization_.apply(n_features, d_token) if bias else None
    
    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return self.weight.shape[0]

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]
    
    def call(self, x):
        x = self.weight[None] * x[..., None]
        print(x.shape)
        if self.bias is not None:
            x = x + self.bias[None]
        print(self.bias[None].shape)
        return x

import numpy as np
x = np.random.rand(4, 5)
n_objects, n_features = x.shape
d_token = 7
tokenizer = NumericalFeatureTokenizer(n_features, d_token, True, 'uniform')
tokens = tokenizer(x)
print("Numerical Token",tokens.shape)

#%%
from typing import List

class CategoricalFeatureTokenizer(tf.keras.layers.Layer):    
    def __init__(
            self,
            cardinalities: List[int],
            d_token: int,
            bias: bool,
            initialization: str,
            ) -> None:
        super(CategoricalFeatureTokenizer, self).__init__()
        
        assert cardinalities, 'cardinalities must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        initialization_ = _TokenInitialization.from_str(initialization)

        self.category_offsets = tf.Variable(tf.cast(tf.cumsum([0]+cardinalities[:-1], axis=0),"float32"), trainable=False)
        self.embeddings = tf.keras.layers.Embedding(sum(cardinalities), d_token, embeddings_initializer=initialization)
        self.bias = initialization_.apply(len(cardinalities), d_token) if bias else None

    def call(self, x):
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x

import numpy as np
cardinalities = [3, 10]
x = np.array([[0, 5],
              [1, 7],
              [0, 2],
              [2, 4]],
             dtype="float32"
             )
n_objects, n_features = x.shape
d_token = 3
tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token, True, 'uniform')
tokens = tokenizer(x)
print("Categorical Token",tokens.shape)