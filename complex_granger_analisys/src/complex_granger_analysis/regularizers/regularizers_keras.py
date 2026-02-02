import tensorflow as tf
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
    alpha=0

    def __init__(self, coeffs=None, enable_cyclic=False):
        if coeffs is None:
            # Domyślnie 20 wartości liniowo od 1 do 3
            coeffs = tf.linspace(1.0, 3.0, 20)
        self.coeffs = tf.convert_to_tensor(coeffs, dtype=tf.float32)
        self.enable_cyclic = bool(enable_cyclic)
        self.period=len(coeffs)

    def __call__(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        abs_x = tf.abs(x)

        # Suma |w| per zmienna wejściowa (oś wejściowa)
        if len(abs_x.shape) == 1:
            # Np. bias: traktujemy każdy element jako osobne "wejście"
            weights_per_input = abs_x
        elif len(abs_x.shape) == 2:
            # Dense kernel: (input_dim, output_dim) -> sum po output_dim
            weights_per_input = tf.reduce_sum(abs_x, axis=1)
        else:
            raise ValueError(f"Nieobsługiwany kształt tensora wag: {abs_x.shape}")

        m = tf.shape(weights_per_input)[0]
        if self.period is None:
            self.period= tf.shape(self.coeffs)[0]

        if self.enable_cyclic:
            indices = tf.math.mod(tf.range(m), self.period)
            multipliers = tf.gather(self.coeffs, indices)
        else:
            multipliers = tf.ones_like(weights_per_input)

        reg = self.alpha*tf.reduce_sum(weights_per_input * multipliers)
        return reg

    def get_config(self):
        # Umożliwia serializację / zapis modelu
        return {
            'alpha': self.alpha,
            "coeffs": self.coeffs.numpy().tolist(),
            "enable_cyclic": self.enable_cyclic,
        }
