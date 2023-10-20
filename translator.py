from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

with open("vectorize.pickle", "rb") as fp:
  data = pickle.load(fp)

eng_vectorizer = TextVectorization.from_config(data["engvec_config"])
eng_vectorizer.set_weights(data["engvec_weights"])
fra_vectorizer = TextVectorization.from_config(data["fravec_config"])
fra_vectorizer.set_weights(data["fravec_weights"])

vocab_size_en = 10000
vocab_size_fr = 20000
seq_length = 20

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



custom_objects = {"PositionalEmbedding": PositionalEmbedding,"CustomSchedule": CustomSchedule, "masked_loss": masked_loss, "masked_accuracy": masked_accuracy}

with tf.keras.utils.custom_object_scope(custom_objects):
  model = tf.keras.models.load_model("model.h5")	
	
def translate(sentence):
  enc_tokens = eng_vectorizer([sentence])
  lookup = list(fra_vectorizer.get_vocabulary())
  start_sentinel, end_sentinel = "[start]", "[end]"
  output_sentence = [start_sentinel]
  for i in range(seq_len):
    vector = fra_vectorizer([" ".join(output_sentence)])
    assert vector.shape == (1, seq_len+1)
    dec_tokens = vector[:, :-1]
    assert dec_tokens.shape == (1, seq_len)
    pred = model([enc_tokens, dec_tokens])
    assert pred.shape == (1, seq_len, vocab_size_fr)
    word = lookup[np.argmax(pred[0, i, :])]
    output_sentence.append(word)
    if word == end_sentinel:
      break
  return output_sentence

def main():
	while True:
		eng_text = input("Write a sentence in English to translate to French, type @exit$ to exit")
		if eng_text == "@exit$":
			break

		fr_text = translate(eng_text)
		print("Translation:", ' '.join(fr_text))

if __name__ == "__main__":
    main()
