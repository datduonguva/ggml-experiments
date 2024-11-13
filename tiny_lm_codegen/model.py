import tensorflow as tf


class Config:
    n_head = 4
    # Todo: adding this
    initializer_range = None



class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, config, scale=False, **kwargs):
        """
        For now, set is_cross_attention = False
        """
        super().__init__(**kwargs)

        self.n_state = nx
        self.n_head = config.n_head

        assert self.n_state % self.n_head == 0

        self.scale = scale
        self.split_size = nx
        self.c_attn = tf.keras.layers.TFConv1D(
            self.n_state * 3,
            self.n_state, 
            initializer_range=config.initializer_range,
            name="c_attn"
        )

    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        training=False,
    ):
        pass
