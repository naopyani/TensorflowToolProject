#!/usr/bin/python
# coding=utf-8
from common_util.data_util import map_sequences
import numpy as np
import tensorflow as tf
import tqdm
import os
import csv


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    vocab_size = 50
    seed = 42
    window_size = 2
    num_ns = 4
    save_targets_path = "./tmp_data/targets.tsv"
    save_contexts_path = "./tmp_data/contexts.tsv"
    save_labels_path = "./tmp_data/labels.tsv"
    read_sequences_path = "./tmp_data/sequences.tsv"
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    with open(read_sequences_path, "r", encoding='utf-8') as sequences, open(save_targets_path, "w",
                                                                             encoding='utf-8') as targets, open(
        save_contexts_path, "w", encoding='utf-8') as contexts, open(save_labels_path, "w",
                                                                     encoding='utf-8') as labels:
        sequence_tqdm = tqdm.tqdm(iter(sequences))
        targets_writer = csv.writer(targets)
        contexts_writer = csv.writer(contexts)
        labels_writer = csv.writer(labels)
        for _corpus in sequence_tqdm:
            sequence = np.array(list(map(int, list(str(_corpus).split("\t")))))
            map_sequences(sequence, num_ns, vocab_size, sampling_table, window_size, seed, targets_writer,
                          contexts_writer, labels_writer)


if __name__ == '__main__':
    main()
