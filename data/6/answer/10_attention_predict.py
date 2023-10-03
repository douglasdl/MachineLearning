#!/usr/bin/env python
# coding: utf-8

from regex import W
import tensorflow as tf
#from tensorflow.keras.layers import GRU, Input, TimeDistributed, Dense, Embedding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from tensorflow.python.keras.utils.vis_utils import plot_model
import pickle
import time
import os

tf.random.set_seed(123)

mode = "predict"
embedding_dim = 256
units = 1024
EPOCHS = 10
BATCH_SIZE = 64
DIR_NAME = "small_parallel_enja/"
checkpoint_dir = './10_attention_checkpoints'

def Preprocess(text):
    text = "<start> "+text+" <end>"
    text = text.replace("\n", " ")
    text = re.sub(r"[' ']+", " ", text)
    return text


def Read(filename):
    new_lines = []
    with open(filename, "r",encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        new_lines.append(Preprocess(line))
    return new_lines


def BuildTokenizer(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='\n', oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def Texts2Sequences(tok, train_texts, valid_texts, test_texts):
    train_sequences = tok.texts_to_sequences(train_texts)
    valid_sequences = tok.texts_to_sequences(valid_texts)
    test_sequences = tok.texts_to_sequences(test_texts)

    padded_train_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        train_sequences, padding="post")

    return padded_train_sequences, valid_sequences, test_sequences


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # スコアを計算するためにこのように加算を実行する
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # スコアを self.V に適用するために最後の軸は 1 となる
        # self.V に適用する前のテンソルの shape は  (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) +
                       self.W2(hidden_with_time_axis)))

        # attention_weights の shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector の合計後の shape == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # アテンションのため
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output の shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # 埋め込み層を通過したあとの x の shape  == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # 結合後の x の shape == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 結合したベクトルを GRU 層に渡す
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

#@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([en_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        # Teacher Forcing - 正解値を次の入力として供給
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # Teacher Forcing を使用
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def evaluate(sentence, input_dim, target_dim):
    #max_length_targ = 20
    #max_length_inp = 18
    attention_plot = np.zeros((target_dim, input_dim))

    inputs = ja_tokenizer.texts_to_sequences([sentence.strip()])

    #inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                           maxlen=input_dim,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([en_tokenizer.word_index['<start>']], 0)

    for t in range(target_dim):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # 後ほどプロットするためにアテンションの重みを保存
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        #result += targ_lang.index_word[predicted_id] + ' '
        result += en_tokenizer.index_word[predicted_id] + ' '

        #if targ_lang.index_word[predicted_id] == '<end>':
        if en_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 予測された ID がモデルに戻される
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def translate(sentence, input_dim, output_dim):
    #result, sentence, attention_plot = evaluate(sentence)
    result, sentence, attention_plot = evaluate(
        sentence, input_dim, output_dim)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(
        result.split(' ')), :len(sentence.split(' '))]
    #plot_attention(attention_plot, sentence.split(' '), result.split(' '))


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

if mode == 'training':
    train_ja = Read(DIR_NAME+"train.ja")
    train_en = Read(DIR_NAME+"train.en")

    valid_ja = Read(DIR_NAME+"dev.ja")
    valid_en = Read(DIR_NAME+"dev.en")

    test_ja = Read(DIR_NAME+"test.ja")
    test_en = Read(DIR_NAME+"test.en")

    ja_tokenizer = BuildTokenizer(train_ja)
    en_tokenizer = BuildTokenizer(train_en)

    train_ja_sequences, valid_ja_sequences, test_ja_sequences = Texts2Sequences(
        ja_tokenizer, train_ja, valid_ja, test_ja)
    train_en_sequences, valid_en_sequences, test_en_sequences = Texts2Sequences(
        en_tokenizer, train_en, valid_en, test_en)

    valid_ja_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        valid_ja_sequences, padding="post")
    valid_en_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        valid_en_sequences, padding="post")
    train_and_valid_ja_sequences = tf.concat(
        [train_ja_sequences, valid_ja_sequences], 0)
    train_and_valid_en_sequences = tf.concat(
        [train_en_sequences, valid_en_sequences], 0)

    dataset = tf.data.Dataset.from_tensor_slices(
        (train_and_valid_ja_sequences, train_and_valid_en_sequences)).shuffle(len(train_and_valid_ja_sequences))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    example_input_batch, example_target_batch = next(iter(dataset))

    BUFFER_SIZE = len(train_and_valid_ja_sequences)
    steps_per_epoch = len(train_and_valid_ja_sequences)//BATCH_SIZE
    vocab_inp_size = len(ja_tokenizer.word_index)+1
    vocab_tar_size = len(en_tokenizer.word_index)+1
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_and_valid_ja_sequences, train_and_valid_en_sequences)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))
    example_input_batch.shape, example_target_batch.shape

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    # サンプル入力
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(
        sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(
        sample_hidden.shape))


    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                      sample_hidden, sample_output)
    print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
        # 2 エポックごとにモデル（のチェックポイント）を保存
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    with open('ja_tokenizer.pickle', 'wb') as handle:
        pickle.dump(ja_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('en_tokenizer.pickle', 'wb') as handle:
        pickle.dump(en_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if mode == 'predict':
    train_ja = Read(DIR_NAME+"train.ja")
    train_en = Read(DIR_NAME+"train.en")

    valid_ja = Read(DIR_NAME+"dev.ja")
    valid_en = Read(DIR_NAME+"dev.en")

    test_ja = Read(DIR_NAME+"test.ja")
    test_en = Read(DIR_NAME+"test.en")

    ja_tokenizer = BuildTokenizer(train_ja)
    en_tokenizer = BuildTokenizer(train_en)

    train_ja_sequences, valid_ja_sequences, test_ja_sequences = Texts2Sequences(
        ja_tokenizer, train_ja, valid_ja, test_ja)
    train_en_sequences, valid_en_sequences, test_en_sequences = Texts2Sequences(
        en_tokenizer, train_en, valid_en, test_en)

    valid_ja_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        valid_ja_sequences, padding="post")
    valid_en_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        valid_en_sequences, padding="post")
    train_and_valid_ja_sequences = tf.concat(
        [train_ja_sequences, valid_ja_sequences], 0)
    train_and_valid_en_sequences = tf.concat(
        [train_en_sequences, valid_en_sequences], 0)

    dataset = tf.data.Dataset.from_tensor_slices(
        (train_and_valid_ja_sequences, train_and_valid_en_sequences)).shuffle(len(train_and_valid_ja_sequences))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    example_input_batch, example_target_batch = next(iter(dataset))

    BUFFER_SIZE = len(train_and_valid_ja_sequences)
    steps_per_epoch = len(train_and_valid_ja_sequences)//BATCH_SIZE
    vocab_inp_size = len(ja_tokenizer.word_index)+1
    vocab_tar_size = len(en_tokenizer.word_index)+1
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_and_valid_ja_sequences, train_and_valid_en_sequences)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))
    example_input_batch.shape, example_target_batch.shape

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    # サンプル入力
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(
        sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(
        sample_hidden.shape))


    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                      sample_hidden, sample_output)
    print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    #checkpoint_dir = './training_checkpoints'
    #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    with open('ja_tokenizer.pickle', 'rb') as handle:
        ja_tokenizer = pickle.load(handle)
    with open('en_tokenizer.pickle', 'rb') as handle:
        en_tokenizer = pickle.load(handle)
    for ja_text, en_text in zip(test_ja, test_en):
        translate(
            ja_text, example_input_batch.shape.dims[1].value, example_target_batch.shape.dims[1].value)
