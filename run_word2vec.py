#!/usr/bin/python
# coding=utf-8
from common_util.data_util import text_line_object
from common_util.data_util import get_sequences_list
from common_util.data_util import generate_training_data
from common_util.data_util import dataset_buffer_batch
from model_util.word2vec_model import Word2Vec
import io
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    data_path = "./data/XXX.txt"
    vocab_size = 4096
    sequence_length = 10
    seed = 42
    batch_size = 1024
    window_size = 2
    num_ns = 4
    buffer_size = 10000
    embedding_dim = 128
    text_ds = text_line_object(data_path)
    sequences, vectorize_layer = get_sequences_list(text_ds, vocab_size, sequence_length, batch_size)
    targets, contexts, labels = generate_training_data(sequences, window_size, num_ns, vocab_size, seed)

    targets = np.array(targets)
    contexts = np.array(contexts)[:, :, 0]
    labels = np.array(labels)

    print('\n')
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")

    dataset = dataset_buffer_batch(targets, contexts, labels, buffer_size, batch_size)
    print(dataset)

    word2vec = Word2Vec(vocab_size, embedding_dim, num_ns)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open('./result/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('./result/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
