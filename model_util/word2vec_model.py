#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
from tensorflow.keras import layers


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=num_ns + 1)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


def train(model, data):
    for t in range(2500):
        bx, by = data.sample(8)
        loss = model.step(bx, by)
        if t % 200 == 0:
            print("step: {} | loss: {}".format(t, loss))


if __name__ == '__main__':
    pass
