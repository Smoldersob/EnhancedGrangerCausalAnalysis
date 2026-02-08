import tensorflow as tf
from tensorflow.keras.layers import Layer

class MaskedDense(Layer):
    def __init__(self, units, kernel_regularizer=None, use_bias=True, 
                 kernel_initializer='zeros', bias_initializer='zeros',
                 forced_relations=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self._user_regularizer = kernel_regularizer
        self.forced_relations = forced_relations
        
    def build(self, input_shape):
        features = input_shape[-1]
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=(features, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        
        self.mask = self.add_weight(
            initializer='ones',
            trainable=False,
            name='mask',
            shape=(features, self.units),
            dtype=tf.float32
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply mask to kernel in forward pass
        masked_kernel = self.kernel * self.mask
        
        # IMPORTANT: Apply forced relations if any
        if self.forced_relations is not None:
            masked_kernel = self._apply_forced_relations(masked_kernel)
        
        output = tf.matmul(inputs, masked_kernel)
        
        if self.use_bias:
            output = output + self.bias
        
        # Add regularization loss for MASKED kernel
        if self._user_regularizer is not None:
            reg_loss = self._user_regularizer(masked_kernel)
            if isinstance(reg_loss, tf.Tensor):
                self.add_loss(reg_loss)
        
        return output
    
    def _apply_forced_relations(self, masked_kernel):
        """Apply forced relation constraints"""
        if not hasattr(self, '_forced_setup_done'):
            self._setup_forced_relations()
        
        if not self.has_forced:
            return masked_kernel
        
        w_selected = tf.gather(masked_kernel, self.i_indices, axis=1)
        mask_selected = tf.gather(self.relation_mask, self.i_indices, axis=1)
        weighted_sums = tf.reduce_sum(w_selected * mask_selected, axis=0)
        
        under_target = tf.cast(weighted_sums < self.values, tf.float32)
        adjustment = tf.expand_dims(
            (tf.abs(weighted_sums - self.values) * under_target) / (weighted_sums + 1e-8), 
            axis=0
        )
        adjustment_values = tf.gather(tf.reshape(adjustment, [-1]), self.relation_ids)
        adjustment_matrix = tf.scatter_nd(self.scatter_indices, adjustment_values, tf.shape(masked_kernel))
        
        return masked_kernel + masked_kernel * adjustment_matrix
    
    def _setup_forced_relations(self):
        """Setup forced relations structures (once)"""
        self.has_forced = self.forced_relations is not None and len(self.forced_relations) > 0
        
        if self.has_forced:
            relation_list = self.forced_relations
            if isinstance(relation_list, tuple):
                relation_list = [relation_list]
            
            self.i_indices = tf.constant([i for i, _, _ in relation_list], dtype=tf.int32)
            self.flat_js = tf.concat([tf.constant(js, dtype=tf.int32) for _, js, _ in relation_list], axis=0)
            self.relation_ids = tf.repeat(tf.range(len(relation_list)), [len(js) for _, js, _ in relation_list])
            self.values = tf.constant([value for _, _, value in relation_list], dtype=tf.float32)
            
            mask_shape = (self.kernel.shape[0], self.kernel.shape[1])
            self.scatter_indices = tf.stack([self.flat_js, tf.gather(self.i_indices, self.relation_ids)], axis=1)
            self.relation_mask = tf.scatter_nd(
                self.scatter_indices, 
                tf.ones_like(self.relation_ids, dtype=tf.float32), 
                shape=mask_shape
            )
        
        self._forced_setup_done = True
    
    def update_mask(self, new_mask):
        """Update mask without recompiling model"""
        self.mask.assign(new_mask)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
        })
        return config

