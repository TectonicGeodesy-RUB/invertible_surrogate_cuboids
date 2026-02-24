# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 13. January 2026
#
# Utility scripts for multisource inversion
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------

import tensorflow as tf
import keras
from keras.utils import register_keras_serializable


@register_keras_serializable()
class BoundedWConstraint_MultiSource(keras.constraints.Constraint):
    """
    Constraint for w of shape (S, 1, 12). Applies row-wise bounds like your single-source constraint.
    Parameter order: [c1,c2,c3,L,W,T,e11,e12,e13,e22,e23,e33]
    """
    def __init__(self, eps=1e-4, max_extent=1.0):
        self.eps = eps
        self.max_extent = max_extent

    def __call__(self, w):
        # w: (S,1,12) -> squeeze to (S,12)
        W = tf.squeeze(w, axis=1)

        eps = tf.constant(self.eps, dtype=W.dtype)
        maxE = tf.constant(self.max_extent, dtype=W.dtype)

        c1 = tf.clip_by_value(W[:, 0:1], -1.0, 1.0)
        c2 = tf.clip_by_value(W[:, 1:2], -1.0, 1.0)
        c3 = tf.clip_by_value(W[:, 2:3], eps, 1.0 - eps)

        L  = tf.clip_by_value(W[:, 3:4], eps, maxE)

        Wmax = 2.0 * tf.minimum(c3 - eps, (1.0 - eps) - c3)
        Wmax = tf.clip_by_value(Wmax, eps, maxE)
        Wdim = tf.clip_by_value(W[:, 4:5], eps, Wmax)

        T  = tf.clip_by_value(W[:, 5:6], eps, maxE)

        strain = W[:, 6:]
        strain = strain / (tf.norm(strain, axis=1, keepdims=True) + tf.constant(1e-12, dtype=W.dtype))

        W_out = tf.concat([c1, c2, c3, L, Wdim, T, strain], axis=1)
        return W_out[:, tf.newaxis, :]  # back to (S,1,12)

    def get_config(self):
        return {"eps": self.eps, "max_extent": self.max_extent}


@register_keras_serializable()
class BoundedWInitializer_MultiSource(keras.initializers.Initializer):
    def __init__(self, n_sources=3, eps=1e-4, max_extent=0.1, c_xy_range=1):
        self.n_sources = n_sources
        self.eps = eps
        self.max_extent = max_extent
        self.c_xy_range = c_xy_range

    def __call__(self, shape, dtype=None):
        rng = tf.random.Generator.from_non_deterministic_state()
        dtype = tf.float32 if dtype is None else dtype
        eps = tf.constant(self.eps, dtype=dtype)
        maxE = tf.constant(self.max_extent, dtype=dtype)

        if len(shape) != 3 or shape[0] != self.n_sources or shape[2] != 12:
            raise ValueError(f"BoundedWInitializer_MultiSource expects shape (n_sources, 1, 12), got {shape}")

        # centroids
        c1 = rng.uniform((self.n_sources, 1), minval=-self.c_xy_range, maxval=self.c_xy_range, dtype=dtype)
        c2 = rng.uniform((self.n_sources, 1), minval=-self.c_xy_range, maxval=self.c_xy_range, dtype=dtype)
        c3 = rng.uniform((self.n_sources, 1), minval=self.eps, maxval=1.0 - self.eps, dtype=dtype)

        # coupled maxima
        Wmax = tf.clip_by_value(2.0 * tf.minimum(c3 - eps, (1.0 - eps) - c3), eps, maxE)

        # sample extents within feasible range
        L = rng.uniform((self.n_sources, 1), minval=self.eps, maxval=maxE, dtype=dtype)
        W = rng.uniform((self.n_sources, 1), minval=self.eps, maxval=Wmax, dtype=dtype)
        T = rng.uniform((self.n_sources, 1), minval=self.eps, maxval=maxE, dtype=dtype)

        # strain
        strain_raw = rng.uniform((self.n_sources, 6), -1.0, 1.0, dtype=dtype)
        strain = strain_raw / (tf.norm(strain_raw, axis=1, keepdims=True) + tf.constant(1e-12, dtype=dtype))

        return tf.concat([c1, c2, c3, L, W, T, strain], axis=1)[:, tf.newaxis, :]

    def get_config(self):
        return {"eps": self.eps, "max_extent": self.max_extent, "c_xy_range": self.c_xy_range}


@register_keras_serializable()
class BoundedMatMulInputLayer_MultiSource(keras.layers.Layer):
    def __init__(self, n_sources: int = 8, use_constrains: bool = False, **kwargs):
        super(BoundedMatMulInputLayer_MultiSource, self).__init__(**kwargs)
        self.n_sources = n_sources
        self.use_constraints = use_constrains

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(self.n_sources, 1, 12),
            initializer=BoundedWInitializer_MultiSource(n_sources=self.n_sources),
            constraint=BoundedWConstraint_MultiSource() if self.use_constraints else None,
            trainable=True,
        )
        super(BoundedMatMulInputLayer_MultiSource, self).build(input_shape)

    def call(self, inputs):
        # self.w: (1, 12) -> transpose: (12, 1)
        w_T = tf.transpose(self.w, (0, 2, 1))
        # inputs: (batch, 12) or (batch, N, 12) depending on case
        out = tf.matmul(inputs, w_T)  # (batch, 1) or (batch, N, 1)
        return tf.squeeze(out, axis=-1)

    def get_config(self):
        config = super(BoundedMatMulInputLayer_MultiSource, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class ScaleLayer_MultiSource(keras.layers.Layer):
    def __init__(self, n_sources, **kwargs):
        super(ScaleLayer_MultiSource, self).__init__(**kwargs)
        self.n_sources = n_sources
        self.scale = self.add_weight(
            name='scale',
            shape=(self.n_sources, 1, 1, 1),  # to (F, 32, 32, 3)
            initializer='ones',
            trainable=True
        )
    def call(self, inputs):
        # inputs: (batch, F, 32, 32, 3)
        scaled = inputs * self.scale  # broadcast scale over spatial and channel dims
        summed = tf.reduce_sum(scaled, axis=1)  # sum over F
        return summed  # shape: (batch, 32, 32, 3)
    def get_config(self):
        config = super(ScaleLayer_MultiSource, self).get_config()
        config.update({'n_sources': self.n_sources})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
