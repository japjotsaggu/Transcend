
#POSIITONAL ENCODING
def pos_enc_matrix(L, d, n=10000):
  assert d % 2 == 0, "Output dimension needs to be an even integer"
  d2 = d//2
  P = np.zeros((L, d))
  k = np.arange(L).reshape(-1, 1) # L-column vector
  i = np.arange(d2).reshape(1, -1) # d-row vector
  denom = np.power(n, -i/d2) # n**(-2*i/d)
  args = k * denom # (L,d) matrix
  P[:, ::2] = np.sin(args)
  P[:, 1::2] = np.cos(args)
  return P

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
    super().__init__(**kwargs)
    self.sequence_length = sequence_length
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim # d_model in paper
    
    # token embedding layer: Convert integer token to D-dim float vector
    self.token_embeddings = tf.keras.layers.Embedding(
    input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
    )
    
    # positional embedding layer: a matrix of hard-coded sine values
    matrix = pos_enc_matrix(sequence_length, embed_dim)
    self.position_embeddings = tf.constant(matrix, dtype="float32")

  def call(self, inputs):
    embedded_tokens = self.token_embeddings(inputs)
    return embedded_tokens + self.position_embeddings

  def compute_mask(self, *args, **kwargs):
    return self.token_embeddings.compute_mask(*args, **kwargs)
  
  def get_config(self):
    config = super().get_config()
    config.update({
    "sequence_length": self.sequence_length,
    "vocab_size": self.vocab_size,
    "embed_dim": self.embed_dim,
    })
    return config

#SELF ATTENTION AND CROSS ATTENTION LAYERS
def self_attention(input_shape, prefix="att", mask=False, **kwargs):
  
  #create layers
  inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32',
  name=f"{prefix}_in1")
  attention = tf.keras.layers.MultiHeadAttention(name=f"{prefix}_attn1", **kwargs)
  norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm1")
  add = tf.keras.layers.Add(name=f"{prefix}_add1")
  
  # functional API to connect input to output
  attout = attention(query=inputs, value=inputs, key=inputs,
  use_causal_mask=mask)
  outputs = norm(add([inputs, attout]))
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}_att")
  return model

def cross_attention(input_shape, context_shape, prefix="att", **kwargs):

  # create layers
  context = tf.keras.layers.Input(shape=context_shape, dtype='float32', name=f"{prefix}_ctx2")
  inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32',name=f"{prefix}_in2")
  attention = tf.keras.layers.MultiHeadAttention(name=f"{prefix}_attn2", **kwargs)
  norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm2")
  add = tf.keras.layers.Add(name=f"{prefix}_add2")
  # functional API to connect input to output
  
  attout = attention(query=inputs, value=context, key=context)
  outputs = norm(add([attout, inputs]))
  # create model and return
  
  model = tf.keras.Model(inputs=[(context, inputs)], outputs=outputs,
  name=f"{prefix}_cross")
  return model

#FEED fORWAD NETWORK
def feed_forward(input_shape, model_dim, ff_dim, dropout=0.1, prefix="ff"):
  # create layers
  inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32',
  name=f"{prefix}_in3")
  dense1 = tf.keras.layers.Dense(ff_dim, name=f"{prefix}_ff1", activation="relu")
  dense2 = tf.keras.layers.Dense(model_dim, name=f"{prefix}_ff2")
  drop = tf.keras.layers.Dropout(dropout, name=f"{prefix}_drop")
  add = tf.keras.layers.Add(name=f"{prefix}_add3")
  
  # functional API to connect input to output
  ffout = drop(dense2(dense1(inputs)))
  norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm3")
  outputs = norm(add([inputs, ffout]))
  
  # create model and return
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}_ff")
  return model

#THE ENCODER 
def encoder(input_shape, key_dim, ff_dim, dropout=0.1, prefix="enc", **kwargs):
  model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=input_shape, dtype='float32', name=f"{prefix}_in0"),
  self_attention(input_shape, prefix=prefix, key_dim=key_dim, mask=False, **kwargs),
  feed_forward(input_shape, key_dim, ff_dim, dropout, prefix)], name=prefix)
  
  return model

#THE DECODER 
def decoder(input_shape, key_dim, ff_dim, dropout=0.1, prefix="dec", **kwargs):
  inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32',name=f"{prefix}_in0")
  context = tf.keras.layers.Input(shape=input_shape, dtype='float32',name=f"{prefix}_ctx0")
  attmodel = self_attention(input_shape, key_dim=key_dim, mask=True,prefix=prefix, **kwargs)
  crossmodel = cross_attention(input_shape, input_shape, key_dim=key_dim,prefix=prefix, **kwargs)
  ffmodel = feed_forward(input_shape, key_dim, ff_dim, dropout, prefix)
  x = attmodel(inputs)
  x = crossmodel([(context, x)])
  output = ffmodel(x)
  
  model = tf.keras.Model(inputs=[(inputs, context)], outputs=output, name=prefix)
  return model

#BUILDING THE TRANSFORMER 
#building a transformer 
def transformer(num_layers, num_heads, seq_len, key_dim, ff_dim, vocab_size_src,vocab_size_tgt, dropout=0.1, name="transformer"):
  embed_shape = (seq_len, key_dim) 
  # set up layers
  
  input_enc = tf.keras.layers.Input(shape=(seq_len,), dtype="int32",name="encoder_inputs")
  input_dec = tf.keras.layers.Input(shape=(seq_len,), dtype="int32",name="decoder_inputs")
  
  embed_enc = PositionalEmbedding(seq_len, vocab_size_src, key_dim, name="embed_enc")
  embed_dec = PositionalEmbedding(seq_len, vocab_size_tgt, key_dim, name="embed_dec")
 
  encoders = [encoder(input_shape=embed_shape, key_dim=key_dim, ff_dim=ff_dim, dropout=dropout, prefix=f"enc{i}",num_heads=num_heads) for i in range(num_layers)]
  decoders = [decoder(input_shape=embed_shape, key_dim=key_dim, ff_dim=ff_dim, dropout=dropout, prefix=f"dec{i}",num_heads=num_heads) for i in range(num_layers)]
  
 #A final Dense layer, final, is created. This layer will produce the model's output 
  final = tf.keras.layers.Dense(vocab_size_tgt, name="linear")
  
  # build output
  x1 = embed_enc(input_enc)
  x2 = embed_dec(input_dec)
  for layer in encoders:
    x1 = layer(x1)
  for layer in decoders:
    x2 = layer([x2, x1])
    output = final(x2)
  
  try:
    del output._keras_mask
  except AttributeError:
    pass
 
  model = tf.keras.Model(inputs=[input_enc, input_dec], outputs=output, name=name)
  return model

#TRAINING 
#using adam optimizer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, key_dim, warmup_steps=4000):
    super().__init__()
    self.key_dim = key_dim
    self.warmup_steps = warmup_steps
    self.d = tf.cast(self.key_dim, tf.float32)
  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d) * tf.math.minimum(arg1, arg2)
  def get_config(self):
    config = {"key_dim": self.key_dim,"warmup_steps": self.warmup_steps}
    return config 

#loss and accuracy 
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
  from_logits=True, reduction='none')
  loss = loss_object(label, pred)
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask
  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss
  
def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred
  mask = label != 0
  match = match & mask
  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

#compiling the transformer with custom learning rate, masked loss and accuracy
vocab_size_en = 10000
vocab_size_fr = 20000
seq_len = 20
num_layers = 4
num_heads = 8
key_dim = 128
ff_dim = 512
dropout = 0.1
model = transformer(num_layers, num_heads, seq_len, key_dim, ff_dim, vocab_size_en, vocab_size_fr, dropout)
lr = CustomSchedule(key_dim)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
model.summary()

history = model.fit(train_ds, epochs=20, validation_data=val_ds)
