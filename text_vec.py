from tensorflow.keras.layers import TextVectorization
with open("text_pairs.pickle", "rb") as tp:
  text_pairs = pickle.load(tp)

random.shuffle(text_pairs)
n_val = int(0.15*len(text_pairs))
n_train = len(text_pairs) - 2*n_val
train_pairs = text_pairs[:n_train]
val_pairs = text_pairs[n_train:n_train+n_val]
test_pairs = text_pairs[n_train+n_val:]

vocab_size_en = 10000
vocab_size_fr = 20000
seq_length = 20

#vectorization
eng_vectorizer = TextVectorization(max_tokens=vocab_size_en,standardize=None,split="whitespace",output_mode="int",output_sequence_length=seq_length,)
fra_vectorizer = TextVectorization(max_tokens=vocab_size_fr,standardize=None,split="whitespace",output_mode="int",output_sequence_length=seq_length + 1)
train_eng_texts = [pair[0] for pair in train_pairs]
train_fra_texts = [pair[1] for pair in train_pairs]
eng_vectorizer.adapt(train_eng_texts)
fra_vectorizer.adapt(train_fra_texts)