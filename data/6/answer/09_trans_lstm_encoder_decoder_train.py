#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('git clone https://github.com/odashi/small_parallel_enja.git')

from regex import W
import tensorflow as tf
from keras.layers import LSTM, Input, TimeDistributed,Dense, Embedding
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import re
from tensorflow.python.keras.utils.vis_utils import plot_model
import pickle

tf.random.set_seed(123)

mode = "learn"
units = 512
epochs = 10
DIR_NAME = "small_parallel_enja/"

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
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='\n', oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def Texts2Sequences(tok, train_texts, valid_texts, test_texts):
    train_sequences = tok.texts_to_sequences(train_texts)
    valid_sequences = tok.texts_to_sequences(valid_texts)
    test_sequences = tok.texts_to_sequences(test_texts)

    padded_train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, padding="post")

    return padded_train_sequences, valid_sequences, test_sequences


if mode == 'learn':
    train_ja = Read(DIR_NAME+"train.ja")
    train_en = Read(DIR_NAME+"train.en")

    valid_ja = Read(DIR_NAME+"dev.ja")
    valid_en = Read(DIR_NAME+"dev.en")

    test_ja = Read(DIR_NAME+"test.ja")
    test_en = Read(DIR_NAME+"test.en")

    ja_tokenizer = BuildTokenizer(train_ja)
    en_tokenizer = BuildTokenizer(train_en)

    train_ja_sequences, valid_ja_sequences, test_ja_sequences = Texts2Sequences(ja_tokenizer, train_ja, valid_ja, test_ja)
    train_en_sequences, valid_en_sequences, test_en_sequences = Texts2Sequences(en_tokenizer, train_en, valid_en, test_en)

if mode == 'predict':
    with open('ja_tokenizer.pickle', 'rb') as handle:
        ja_tokenizer = pickle.load(handle)
    with open('en_tokenizer.pickle', 'rb') as handle:
        en_tokenizer = pickle.load(handle)


def TrainModel(ja_vocab_size, en_vocab_size):
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_emb = Embedding(ja_vocab_size,units,mask_zero=True, name="encoder_emb")(encoder_inputs)
    encoder_lstm = LSTM(units,return_state=True, name="encoder_lstm")
    encoder_outputs,state_h, state_c = encoder_lstm(encoder_emb)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_emb = Embedding(en_vocab_size, units, mask_zero=True, name="decoder_emb")(decoder_inputs)

    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="decoder_lstm")

    decoder_outputs, _, _ = decoder_lstm(decoder_emb,initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(en_vocab_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),)
    model.summary()
    return model

def SaveModel(ja_tokenizer,en_tokenizer,model):
    with open('ja_tokenizer.pickle', 'wb') as handle:
        pickle.dump(ja_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('en_tokenizer.pickle', 'wb') as handle:
        pickle.dump(en_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for layer in model.layers:
        weights = layer.get_weights()
        if weights != []:
            np.savez(f'{layer.name}.npz', np.array(weights, dtype=object)) #np.savez(f'{layer.name}.npz', weights)

def LoadModel(model):
    for layer in model.layers:
        try:
            data = np.load(f'{layer.name}.npz', allow_pickle=True)
            layer.set_weights(data['arr_0'])
        except FileNotFoundError as e:
            print(e)

def PredictModel(ja_vocab_size, en_vocab_size):
    # encoderのモデルを構築
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_emb = Embedding(ja_vocab_size, units,mask_zero=True, name="encoder_emb")(encoder_inputs)
    encoder_lstm = LSTM(units,return_state=True, name="encoder_lstm")
    encoder_outputs,state_h, state_c = encoder_lstm(encoder_emb)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_state_in = [Input(shape=(units,)), Input(shape=(units,))]
    decoder_emb = Embedding(en_vocab_size, units,mask_zero=True, name="decoder_emb")(decoder_inputs)
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_emb,initial_state=decoder_state_in)
    decoder_states = [decoder_state_h,decoder_state_c]
    decoder_dense = TimeDistributed(Dense(en_vocab_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_state_in,[decoder_outputs] + decoder_states) # リストを+で結合
    encoder_model.summary()
    decoder_model.summary()

    return encoder_model,decoder_model


#評価方法はBLEU
def Predict(encoder_model,decoder_model,ja_text):
    sequence = ja_tokenizer.texts_to_sequences([ja_text.strip()])
    sequence = np.array(sequence)
    encoder_outputs = encoder_model.predict(sequence)
    decoder_hidden_states = encoder_outputs
    decoder_inputs = np.array(en_tokenizer.texts_to_sequences([["<start>"]]))
    ans = "<start>"
    for _ in range(20):
        prediction,decoder_hidden_state_h, decoder_hidden_state_c = decoder_model.predict([decoder_inputs]+decoder_hidden_states)
        decoder_hidden_states = [ decoder_hidden_state_h, decoder_hidden_state_c]
        prediction_id = tf.argmax(prediction, axis=-1)[0][0].numpy()
        word = en_tokenizer.sequences_to_texts([[prediction_id]])
        decoder_inputs = np.reshape(np.array(prediction_id), [-1, 1])
        ans += " "+word[0]
        if word[0] == "<end>":
            break
    return ans

def Evaluate(encoder_model,decoder_model,ja_texts,en_texts):
    for ja_text, en_text in zip(ja_texts, en_texts):
        result = Predict(encoder_model,decoder_model,ja_text)
        print(ja_text,en_text,result)


if mode == "learn":
    valid_ja_sequences = tf.keras.preprocessing.sequence.pad_sequences(valid_ja_sequences, padding="post")
    valid_en_sequences = tf.keras.preprocessing.sequence.pad_sequences(valid_en_sequences, padding="post")

    train_and_valid_ja_sequences = tf.concat([train_ja_sequences, valid_ja_sequences], 0)
    train_and_valid_en_sequences = tf.concat([train_en_sequences, valid_en_sequences], 0)
    model = TrainModel(len(ja_tokenizer.word_index)+1, len(en_tokenizer.word_index)+1)
    #plot_model(model, show_shapes=True, show_layer_names=False)
    model.fit([train_and_valid_ja_sequences, train_and_valid_en_sequences[:, :-1]],train_and_valid_en_sequences[:, 1:], batch_size=32, epochs=epochs)
    SaveModel(ja_tokenizer,en_tokenizer,model)

if mode == 'predict':
    encoder, decoder = PredictModel(len(ja_tokenizer.word_index)+1, len(en_tokenizer.word_index)+1)
    LoadModel(encoder)
    LoadModel(decoder)
    test_ja = Read(DIR_NAME+"test.ja")
    test_en = Read(DIR_NAME+"test.en")
    Evaluate(encoder,decoder,test_ja,test_en)

