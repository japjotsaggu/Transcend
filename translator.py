import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
import pickle
from source import PositionalEmbedding, CustomSchedule, masked_loss, masked_accuracy
from text_vec import eng_vectorizer, fra_vectorizer


vocab_size_en = 10000
vocab_size_fr = 20000
seq_length = 20

custom_objects = {"PositionalEmbedding": PositionalEmbedding, "CustomSchedule": CustomSchedule, "masked_loss": masked_loss, "masked_accuracy": masked_accuracy}

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
