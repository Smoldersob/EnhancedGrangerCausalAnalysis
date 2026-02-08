import tensorflow as tf
import numpy as np
from tensorflow.keras.regularizers import Regularizer

class KerasCyclicL1Regularizer(Regularizer):
    """
    L1 regularizer with cyclingly growing weights.
    Sum of |w| is multiply by coefficient from coeffs wecto.
    Cycling: indexrs of weights are are looped ever period of elements form coeffs
    vector.
    Regularizer is dedicated for models, where inputs from one period are of simiular
    nature but are order by probability of effect (ex. autoregression, power series). 
    """
    l1 = 0

    def __init__(self, coeffs=None, enable_cyclic=False):
        if coeffs is None:
            coeffs = tf.linspace(1.0, 3.0, 20)
        self.coeffs = tf.convert_to_tensor(coeffs, dtype=tf.float32)
        self.enable_cyclic = bool(enable_cyclic)
        self.lag_order = len(coeffs)
        self.indices = None

    def set_lag_orders(self, lag_order):
        self.indices = tf.constant(np.concatenate([np.arange(l) for l in np.diff(lag_order)]), dtype=tf.int32)

    def __call__(self, x):
        if self.l1==0 or self.l1 is None:
            return tf.constant(0.0, dtype=tf.float32)
        abs_x = tf.abs(x)

        if len(abs_x.shape) == 1:
            weights_per_input = abs_x
        elif len(abs_x.shape) == 2:
            weights_per_input = tf.reduce_sum(abs_x, axis=1)
        else:
            raise ValueError(f"Unsupported weight tensor shape: {abs_x.shape}")

        if self.enable_cyclic and self.indices is not None:
            multipliers = tf.gather(self.coeffs, self.indices)
        else:
            multipliers = 1.0

        return self.l1 * tf.reduce_sum(weights_per_input * multipliers)

    def get_config(self):
        # Umożliwia serializację / zapis modelu
        return {
            'l1': self.l1,
            "coeffs": self.coeffs.numpy().tolist(),
            "enable_cyclic": self.enable_cyclic,
        }
