import tensorflow as tf


def stable_softmax(logits, axis=None, name=None):
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)


class GPT2Config:
    n_head = 4
    n_layers = 4
    n_inner = None # this should not change
    n_embd = 128
    vocab_size = 50257
    n_embd = 128  # Same with n_embedding ? need verify
    n_positions = 512
    layer_norm_epsilon = 1e-5
    # Todo: adding this
    output_attentions = False
    output_hidden_states = False

    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-05
    initializer_range = 0.02
    use_cache = False
    bos_token_id = 50256
    eos_token_id = 50256

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
            nx=nx,
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
                current_shape[2],
                current_shape[3]
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


class TFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(
            n_state, nx, initializer_range=config.initializer_range, name="c_fc"
        )
        self.c_proj = TFConv1D(
            nx, n_state, initializer_range=config.initializer_range, name="c_proj"
        )
        self.act = tf.keras.activations.relu
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.intermediate_size = n_state
        self.embed_dim = nx

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "c_fc", None) is not None:
            with tf.name_scope(self.c_fc.name):
                self.c_fc.build([None, None, self.intermediate_size])
        if getattr(self, "c_proj", None) is not None:
            with tf.name_scope(self.c_proj.name):
                self.c_proj.build([None, None, self.embed_dim])



class TFBlock(tf.keras.layers.Layer):
    def __init__(self, config, scale=False, **kwargs):
        super().__init__(**kwargs)

        nx = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
        self.ln_1 = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name='ln_1'
        )
        self.attn = TFAttention(nx, config, scale, name='attn')
        self.ln_2 = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name='ln_2'
        )

        self.mlp = TFMLP(inner_dim, config, name='mlp')
        self.n_embd = config.n_embd


    def call(
        self, 
        x,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache=False,
        output_attentions=False,
        training=False
    ):
        a = self.ln_1(x)
        output_attn = self.attn(
            a,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training
        )

        a = output_attn[0] 
        outputs = outputs[1:]
        x = x + a
        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        return [x] + outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "ln_1", None) is not None:
            with tf.name_scope(self.ln_1.name):
                self.ln_1.build([None, None, self.n_embd])
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        if getattr(self, "ln_2", None) is not None:
            with tf.name_scope(self.ln_2.name):
                self.ln_2.build([None, None, self.n_embd])
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
        if getattr(self, "ln_cross_attn", None) is not None:
            with tf.name_scope(self.ln_cross_attn.name):
                self.ln_cross_attn.build([None, None, self.n_embd])


class TFGPT2(tf.keras.models.Model):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        print("TFGPT2 initiated")

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = False  # default to False for now
        self.num_hidden_layers = config.n_layers
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        # word token embedding
        self.wte = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.n_embd,
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

        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [
            TFBlock(config, scale=True, name=f"h_._{i}")
            for i in range(config.n_layers)
        ]

        self.ln_f = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_f"
        )
        self.embed_dim = config.n_embd

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

        if attention_mask is not None:
            shape = tf.shape(attention_mask)
            attention_mask = tf.reshape(
                attention_mask,
                [shape[0], 1, 1, shape[1]]
            )
            one_cst = tf.constant(1.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multily(
                tf.subtract(one_cst, attention_mask),
                tf.constant(-10000)
            )

        position_ids = tf.reshape(
            position_ids, 
            [-1, tf.shape(position_ids)[-1]]
        )


        inputs_embeds = self.wte(input_ids) # (batch, length, n_embd)
        position_embeds = self.wpe(input_ids) # (batch, length, n_embd)
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

            hidden_states, present = outputs[:2]

        hidden_states = self.ln_f(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)


        return hidden_states


    def build(self, input_shape=None):
        """
        The build method helps make the models 
        """
        if self.built:
            return 
        self.built = True
        
        if getattr(self, "wte", None) is not None:
            with tf.name_scope(self.wte.name):
                self.wte.build(None)

        if getattr(self, "wpe", None) is not None:
            with tf.name_scope(self.wpe.name):
                self.wpe.build(None)

        if getattr(self, "ln_f", None) is not None:
            with tf.name_scope(self.ln_f.name):
                self.ln_f.build([None, None, self.n_embd])
        if getattr(self, "h", None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFGPT2LMHeadModel(tf.keras.models.Model):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPT2(config, name="transformer")
        
    
    def hf_compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=keras.losses.Reduction.NONE
        )

        # Clip negative labels to zero here to avoid NaNs and errors - 
        # those positions will get masked later anyway
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # make sure only labels that are not equal to -100 affect the loss
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))


    def call(
        self,
        input_ids,
        past_key_values,
        attention_mask,
        token_type_ids,
        position_ids,
        use_cache,
        labels=None,
        training=False
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            training=training
        ) 

        hidden_states = transformer_outputs[0]

        logits = tf.matmul(
            hidden_states, self.transformer.wte.weights,transpose_b = True
        )

        loss = None
        if labels is not None:
            # label is the exact inputs, so we should ignore the last token
            # of the logits, and first token of the inputs
            shifted_logits = logits[:, :-1] 
            labels = labels[:, 1:]
            loss = self.hf_compute_loss(labels, shifted_logits)

    def build(self, input_shape=None):
        if self.built:
            return True

        self.built = True

        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)

            

if __name__ == '__main__':
    config = GPT2Config()
    m = TFGPT2LMHeadModel(config, name="transformer")
    m.build()
    m.summary()
