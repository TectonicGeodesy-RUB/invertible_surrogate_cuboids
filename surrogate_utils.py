# --------------------------------------------------------------
# Modified by Kaan Cökerim¹ on 01. December 2025
# Original version created by Jonathan Bedford¹ on 10 October 2025
#
# Utility script for training and inversion
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------
import tensorflow as tf
from keras.utils import register_keras_serializable
import keras
import numpy as np
import datetime
import time

# ------------- CALLBACKS
class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        logs = logs or {}
        logs['learning_rate'] = lr


class VerboseCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_epoch_loss = np.inf
        self.train_start_time = None
        self.last_epoch_time = None
        self.total_epochs = None

    def on_train_begin(self, logs=None):
        self.best_epoch_loss = np.inf

        self.train_start_time = time.time()
        self.last_epoch_time = self.train_start_time

        self.total_epochs = self.params.get("epochs", None)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Format seconds as H:MM:SS or M:SS."""
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # ---------- Time bookkeeping ----------
        now = time.time()
        epoch_time = now - self.last_epoch_time
        self.last_epoch_time = now

        elapsed = now - self.train_start_time
        epochs_done = epoch + 1
        avg_epoch_time = elapsed / max(epochs_done, 1)

        if self.total_epochs is not None:
            remaining_epochs = max(self.total_epochs - epochs_done, 0)
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_str = datetime.timedelta(seconds=eta_seconds)
        else:
            eta_str = "n/a"

        epoch_time_str = datetime.timedelta(seconds=epoch_time)

        # ---------- Learning rate ----------
        opt = self.model.optimizer
        lr = opt.learning_rate
        if hasattr(lr, '__call__'):
            step = tf.cast(opt.iterations, tf.float32)
            lr = lr(step)
        lr = float(tf.keras.backend.get_value(lr))

        # ---------- Best val_loss & emoji ----------
        change_emoji = "🟢" if logs['val_loss'] < self.best_epoch_loss else "🔴"
        if logs['val_loss'] < self.best_epoch_loss:
            self.best_epoch_loss = logs['val_loss']

        # ---------- Logging line ----------
        timestamp = datetime.datetime.now().strftime('%d-%b %H:%M:%S')
        metrics_str = " | ".join([f"{k}: {v:.4e}" for k, v in logs.items()])

        print(
            f"+\t[{timestamp}] - Epoch {epoch + 1:05d}: "
            f"{metrics_str} {change_emoji} | "
            f"lr: {lr:.4e} | "
            f"epoch_time: {epoch_time_str} | "
            f"ETA: {eta_str}"
        )


# ------------- CUSTOM CLASSES
@register_keras_serializable()
class Apply_LV_range(keras.layers.Layer):
    def __init__(self, input_shape=None, **kwargs):
        super(Apply_LV_range, self).__init__(**kwargs)
        self.input_shape_ = input_shape
    def call(self, x_in, logvar_range, var_weight):
        x = tf.reduce_mean(logvar_range) + 0.5 * (tf.reduce_max(logvar_range)-tf.reduce_min(logvar_range)) * x_in
        lv_loss = tf.reduce_mean(-1 * x_in) * var_weight
        self.add_loss(lv_loss)
        return x
    def compute_output_shape(self):
        return (None, 12)
    def get_config(self):
        config = super(Apply_LV_range, self).get_config()
        config.update({"input_shape": self.input_shape_})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class Sampled(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Sampled, self).__init__(**kwargs)
    def call(self, x, log_variance):
        # Sample from a normal distribution with mean 0 and standard deviation exp(log_variance/2)
        epsilon = tf.random.normal(tf.shape(x))
        sampled = x + tf.exp(0.5*log_variance) * epsilon
        return sampled
    def get_config(self):
        config = super(Sampled, self).get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class ScaleLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.scale = self.add_weight(name='scale', initializer='ones')
    def call(self, inputs):
        return inputs * self.scale
    def get_config(self):
        config = super(ScaleLayer, self).get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class MatMulInputLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MatMulInputLayer, self).__init__(**kwargs)
        self.w = self.add_weight(name='w', shape=(1, 12), initializer='random_normal', trainable=True)
    def call(self, inputs):
        w_T = tf.transpose(self.w)  # (8, 1)
        out = tf.matmul(inputs, w_T)  # (batch, 8, 1)
        return tf.squeeze(out, axis=-1)  # (batch, 8)
    def get_config(self):
        config = super(MatMulInputLayer, self).get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class BoundedWInitializer(keras.initializers.Initializer):
    def __init__(self, eps=1e-4, max_extent=0.1, c_xy_range=1):
        self.eps = eps
        self.max_extent = max_extent
        self.c_xy_range = c_xy_range

    def __call__(self, shape, dtype=None):
        rng = tf.random.Generator.from_non_deterministic_state()
        dtype = tf.float32 if dtype is None else dtype
        eps = tf.constant(self.eps, dtype=dtype)
        maxE = tf.constant(self.max_extent, dtype=dtype)

        if len(shape) != 2 or shape[0] != 1 or shape[1] != 12:
            raise ValueError(f"BoundedWInitializer expects shape (1, 12), got {shape}")

        # centroids
        c1 = rng.uniform((1, 1), minval=-self.c_xy_range, maxval=self.c_xy_range, dtype=dtype)
        c2 = rng.uniform((1, 1), minval=-self.c_xy_range, maxval=self.c_xy_range, dtype=dtype)
        c3 = rng.uniform((1, 1), minval=self.eps, maxval=1.0 - self.eps, dtype=dtype)

        # coupled maxima
        Wmax = tf.clip_by_value(2.0 * tf.minimum(c3 - eps, (1.0 - eps) - c3), eps, maxE)

        # sample extents within feasible range
        L = rng.uniform((1, 1), minval=self.eps, maxval=maxE, dtype=dtype)
        W = rng.uniform((1, 1), minval=self.eps, maxval=Wmax, dtype=dtype)
        T = rng.uniform((1, 1), minval=self.eps, maxval=maxE, dtype=dtype)

        # strain direction
        strain_raw = rng.uniform((1, 6), -1.0, 1.0, dtype=dtype)
        strain = strain_raw / (tf.norm(strain_raw, axis=1, keepdims=True) + tf.constant(1e-12, dtype=dtype))

        return tf.concat([c1, c2, c3, L, W, T, strain], axis=1)

    def get_config(self):
        return {"eps": self.eps, "max_extent": self.max_extent, "c_xy_range": self.c_xy_range}


@register_keras_serializable()
class BoundedWConstraint(keras.constraints.Constraint):
    def __init__(self, eps=1e-4, max_extent=1.0):
        self.eps = eps
        self.max_extent = max_extent

    def __call__(self, w):
        # w.shape: (1, 12)
        eps = tf.constant(self.eps, dtype=w.dtype)
        max_extent = tf.constant(self.max_extent, dtype=w.dtype)

        # centroids
        c1 = tf.clip_by_value(w[:, 0:1], -1.0, 1.0)
        c2 = tf.clip_by_value(w[:, 1:2], -1.0, 1.0)
        c3 = tf.clip_by_value(w[:, 2:3], eps, 1.0 - eps)

        # cuboid dimensions
        Wmax = tf.minimum(2.0*(c3 - eps), max_extent - eps) # 2.0 * tf.minimum(c3 - eps, (1.0 - eps) - c3)
        Wmax = tf.clip_by_value(Wmax, eps, max_extent)

        # Apply bounds
        L = tf.clip_by_value(w[:, 3:4], eps, max_extent)
        W = tf.clip_by_value(w[:, 4:5], eps, Wmax)
        T = tf.clip_by_value(w[:, 5:6], eps, max_extent)

        # --- strain direction: enforce ||e||_2 = 1 ---
        strain = w[:, 6:]
        strain = strain / (tf.norm(strain, axis=1, keepdims=True) + tf.constant(1e-12, dtype=w.dtype))

        return tf.concat([c1, c2, c3, L, W, T, strain], axis=1)

    def get_config(self):
        return {"eps": self.eps, "max_extent": self.max_extent}


@register_keras_serializable()
class BoundedMatMulInputLayer(keras.layers.Layer):
    def __init__(self, use_constrains: bool = False, bounded_init: bool = False, **kwargs):
        super(BoundedMatMulInputLayer, self).__init__(**kwargs)
        self.use_constraints = use_constrains
        self.bounded_init = bounded_init

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(1, 12),
            initializer=BoundedWInitializer() if self.bounded_init else None,
            constraint=BoundedWConstraint() if self.use_constraints else None,
            trainable=True,
        )
        super(BoundedMatMulInputLayer, self).build(input_shape)

    def call(self, inputs):
        # self.w: (1, 12) -> transpose: (12, 1)
        w_T = tf.transpose(self.w)
        # inputs: (batch, 12)
        out = tf.matmul(inputs, w_T)  # (batch, 1)
        return tf.squeeze(out, axis=-1)

    def get_config(self):
        config = super(BoundedMatMulInputLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def surrogate_model() -> keras.Model:
    """
    Functional generation of the surrogate model architecture.
    Returns:
        keras Model with the surrogate model architecture.
    """

    # Hyperparameters for log variance bounding and loss weighting
    var_weight = tf.constant(1e-4)
    std_range = tf.cast(tf.constant([0.001, 0.05]), 'float32')
    logvar_range = 2 * tf.math.log(std_range)

    # Model input
    inputs = keras.Input(shape=(12,))

    x_logvar = keras.layers.Dense(12)(inputs)
    x_logvar = keras.layers.LeakyReLU()(x_logvar)
    x_logvar = keras.layers.BatchNormalization()(x_logvar)

    x_logvar = keras.layers.Dense(12)(x_logvar)
    x_logvar = keras.layers.LeakyReLU()(x_logvar)
    x_logvar = keras.layers.BatchNormalization()(x_logvar)

    x_logvar = keras.layers.Dense(12)(x_logvar)
    x_logvar = keras.layers.LeakyReLU()(x_logvar)
    x_logvar = keras.layers.BatchNormalization()(x_logvar)

    x_logvar = keras.layers.Dense(12, activation='tanh')(x_logvar)
    x_logvar = Apply_LV_range()(x_logvar, logvar_range, var_weight)

    # Sampling
    x_sampled = Sampled()(inputs, x_logvar)

    # vec2image branch
    x = keras.layers.Dense(128)(x_sampled)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(8 * 8 * 64)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Reshape((8, 8, 64))(x)

    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    # Conv2DTranspose up-scaling
    x = keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)  # 8x8 -> 16x16
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2DTranspose(16, kernel_size=6, strides=2, padding='same')(x)  # 16x16 -> 32x32
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2DTranspose(8, kernel_size=10, strides=2, padding='same')(x)  # 32x32 -> 64x64
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    outputs = keras.layers.Conv2D(3, kernel_size=12, padding='same', activation='linear')(x)


    model = keras.Model(inputs=inputs, outputs=[outputs, x_logvar, x_sampled])
    model.name = 'v2i_max_entropy_C2DT_LAPLACE'
    model.summary()

    model.compile(optimizer='adam', loss=['mse', None, None])

    return model