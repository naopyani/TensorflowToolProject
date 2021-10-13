#!/usr/bin/python
# coding=utf-8
from common_util.data_util import get_int_bag, get_int_list, dataset_buffer_batch
from model_util.word2vec_model import Word2Vec
import io
import numpy as np
import tensorflow as tf
import os
import pandas as pd


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    vocab_size = 50
    batch_size = 2
    epochs = 3
    num_ns = 4
    buffer_size = 10000
    embedding_dim = 128

    save_targets_path = "./tmp_data/targets.tsv"
    save_contexts_path = "./tmp_data/contexts.tsv"
    save_labels_path = "./tmp_data/labels.tsv"
    vocab_path = "./tmp_data/vocab.tsv"
    word2vec_path = "./tmp_data/word2vectors.tsv"
    targets_array = np.array(get_int_bag(save_targets_path))
    print(targets_array.shape)
    contexts_array = np.array(get_int_list(save_contexts_path))
    print(contexts_array.shape)
    labels_array = np.array(get_int_list(save_labels_path))
    print(labels_array.shape)

    dataset = dataset_buffer_batch(targets_array, contexts_array, labels_array, buffer_size, batch_size)
    print(dataset)
    del targets_array
    del contexts_array
    del labels_array

    word2vec = Word2Vec(vocab_size, embedding_dim, num_ns)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=epochs)
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

    df = open(vocab_path, "r", encoding='utf-8')
    corpus = df.readlines()
    df.close()
    vocab = list(map(lambda y: str(y).replace("\n", ""), corpus))
    word_col = pd.DataFrame(vocab[2:], columns=['word'])
    vec_col = pd.DataFrame(weights[2:len(vocab[2:]) + 2])
    result_w2v = pd.concat([word_col, vec_col], axis=1)
    result_w2v.to_csv(word2vec_path, index=False)


if __name__ == '__main__':
    main()
