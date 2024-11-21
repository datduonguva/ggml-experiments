import tensorflow as tf


def stable_softmax(logits, axis=None, name=None):
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)


class Config:
    n_head = 4
    n_layers = 4
    n_embedding = 512
    hidden_size = 512  # Same with n_embedding ? need verify
    n_positions = None
    layer_norm_epsilon = 1e-5
    # Todo: adding this
    initializer_range = None

    output_attentions = False


class TFConv1D(tf.keras.layers.Layer):
    """
    This seems like it is exactly the same with the tf.keras.layers.Dense
    """
    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        """
        nf: the output dimension
        nx: the input dimension
        """
        super().__init__(**kwargs)

        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        self.weight = self.add_weight(
            "weight",
            shape=[self.nx, self.nf],
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
        )
        self.bias = self.add_weight(
            "bias", shape=[1, self.nf], initializer=tf.zeros_initializer()
        )

        def call(self, x):
            bz, sl = tf.shape(x)[:2]

            x = tf.reshape(x, [-1, self.nx])
            x = tf.matmul(x, self.weight) + self.bias

            x = tf.reshape(x, [bz, sl, self.nf])

            return x


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
        self.c_attn = TFConv1D(
            nf=self.n_state * 3,
            nx=self.n_state,
            initializer_range=config.initializer_range,
            name="c_attn",
        )

        self.c_proj = TFConv1D(
            nf=self.n_state,
            nx=self.nx,
            initializer_range=config.initializer_range,
            name="c_proj"
        )

        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.embed_dim = self.n_state

    def build(self, input_shape=None):
        """
        This is to be able build the model without compiling, as the same is
        known
        """
        if self.built:
            return
        self.build = True

        c_attn_shape = 3 * self.embed_dim

        if getattr(self, "c_proj") is not None:
            with tf.name_scope(self.c_proj.name):
                self.c_proj.build([None, None, self.embed_dim])
        if getattr(self, "c_attn") is not None:
            with tf.name_scope(self.c_attn.name):
                self.c_attn.build([None, None, self.embed_dim])

    def split_heads(self, x):
        """
        x : (batch, length, n_embedding)
        """
        x_shape = tf.shape(x) 
        x = tf.reshape(
            x,
            [x_shape[0], x_shape[1], self.n_head, x_shape[-1]/self.n_head]
        )   # (batch, length, n_head, head_size)

        x = tf.transpose(x, (0, 2, 1, 3)) # (batch, n_head, length, head_size)

        return x


    def _attn(
        self, q, k, v, attention_mask, head_mask, output_attentions,
        training=False
    ):
        # q, k, v are after split -> (batch, n_head, length, features)
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(tf.shape(k)[-1], dtype=w.dtype)
            w = w / tf.math.sqrt(dk)

        # not using cross attention
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, w.dtype)
            w = w + attention_mask 

        w = stable_softmax(logits=w, axis=-1)
        w = self.attn_dropout(w, training=training)

        if head_mask is not None:
            w = w*head_mask

        output = [tf.matmul(w, v)]

        if output_attentions:
            outputs.append(w)
        return outputs


    def merge_heads(self, x):
        """
        x has shape (batch, n_heads, length, features)
        """


        x = tf.transpose(x, [0, 2, 1, 3])
        current_shape = tf.shape(x)
        x = tf.reshape(
            x,
            [
                current_shape[0],
                current_shape[1],
                current_shape[2]current_shape[3]
            ]
        )
        return x


    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        use_cache,
        training=False,
    ):
        """
        x.shape = (batch, length, embedding_size)
        """
        # not using cross attention here
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)


        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0, num=2)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value], value, axis=-2)

        if use_cache:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None, )


        attn_outputs = self._attn(
            query, key, value, 
            attention, head_mask, 
            training=training 
        )

        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)


        outputs = [a, present] + attn_outputs[1:]

        return outputs # (a, present) + attentions




class TFGPT2(tf.keras.models.Model):
    def __int__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = False  # default to False for now
        self.num_hidden_layers = config.n_layers
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        # word token embedding
        self.wpe = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                config.initializer_range
            ),
            name="wte",
        )
       
        # word position embedding 
        self.wpe = tf.keras.layers.Embedding(
            input_dim=config.n_positions,
            output_dim=config.n_embd,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                config.initializer_range
            ),
            name="wpe",
        )

        self.drop = tf.keras.layers.Drop(config.embd_drop)
        self.h = [
            TFBlock(config, scale=True, name=f"h_._{i}")
            for i in range(config.n_layers)
        ]

        self.ln_f = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_f"
        )
        self.embed_dim = config.hidden_size

    def call(self, 
        input_ids,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        training=False
    ):
        input_shape = tf.shape(input_ids)
        input_ids = tf.reshape(
            input_ids,
            [-1, input_shape[-1]]
        )


        if past_key_values is None:
            past_length = 0
            past_key_values = [None]*len(self.h)
        else:
            past_length = tf.shape(
                past_key_values[0][0]
            )[-2]

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(past_length, input_shape[-1] + past_length),
                axis=0
            )

        if attention_mask is not None
            shape = tf.shape(attention_mask)
            attention_mask = tf.reshape(
                attention_mask,
                [shape[0], 1, 1, shape[1]]
            )
            one_cst = tf.constant(1.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multily(
                tf.subtract(one_cst, attention_mask)
                tf.constant(-10000)
            )

        position_ids = tf.reshape(
            position_ids, 
            [-1, tf.shape(position_ids)[-1]]
        )


        inputs_embeds = self.wte(input_ids) # (batch, length, hidden_size)
        position_embeds = self.wpe(input_ids) # (batch, length, hidden_size)
        token_type_embeds = tf.constant(0.0, dtype=inputs_embeds.dtype)

        position_embeds = tf.cast(position_embeds, dtype=inputs_embeds.dtype)

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [tf.shape(hidden_states)[-1]]

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=None,
                output_attentions=None,
                training=training
            )
            )

        
