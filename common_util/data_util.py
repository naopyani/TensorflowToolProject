#!/usr/bin/python
# coding=utf-8
import os
from urllib.request import urlretrieve
import zipfile
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import jieba.posseg
import jieba.analyse


def get_corpus(word_pre_path):
    df = open(word_pre_path, "r", encoding='utf-8')
    corpus = df.readlines()
    df.close()
    return corpus


def download_dataset(source_url, save_path):
    if not os.path.exists(save_path):
        print("Downloading the dataset... (It may take some time)")
        filename, _ = urlretrieve(source_url, save_path)
        print("Done!")


def get_zip_word_list(data_path):
    with zipfile.ZipFile(data_path) as f:
        text_words = f.read(f.namelist()[0]).lower().split()
    return text_words


def get_word_list(word_pre_path):
    corpus = get_corpus(word_pre_path)
    raw_word = list(map(lambda y: list(str(y).replace("\n", "").split(" ")), corpus))
    return raw_word


def get_int_list(word_pre_path):
    corpus = get_corpus(word_pre_path)
    filter_corpus = list(filter(lambda x: x != '\n', corpus))
    raw_word = list(map(lambda y: list(map(int, str(y).replace("\n", "").split(","))), filter_corpus))
    return raw_word


def get_int_bag(word_pre_path):
    corpus = get_corpus(word_pre_path)
    filter_corpus = list(filter(lambda x: x != '\n', corpus))
    word_bag = list(map(lambda x: int(str(x).replace("\n", "")), filter_corpus))
    return word_bag


def text_line_object(data_path):
    """

    :param data_path:读取样本文件位置
    :return:
    """
    text_ds = tf.data.TextLineDataset(data_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    return text_ds


def get_sequences_list(text_ds, vocab_size, sequence_length, batch_size):
    """
    把文本矢量规范化标准化到整数，样本数据每行都有相同长度
    :param batch_size:
    :param text_ds: 每行的样本对象
    :param vocab_size: 词汇量
    :param sequence_length: 句子固定长度
    :return: 序列化后的文本list
    """

    # Use the TextVectorization layer to normalize, split, and map strings to
    # integers. Set output_sequence_length length to pad all samples to same length.
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)
    # inverse_vocab = vectorize_layer.get_vocabulary()
    vectorize_layer.adapt(text_ds.batch(batch_size))
    # text向量化
    text_vector_ds = text_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())
    vocab = vectorize_layer.get_vocabulary()
    print("词汇量" + str(len(vocab)))
    return sequences, vocab


# Iterate over all sequences (sentences) in dataset.
def map_sequences(sequence, num_ns, vocab_size, sampling_table, window_size, seed, targets_writer, contexts_writer,
                  labels_writer):
    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size=vocab_size,
                                                                       sampling_table=sampling_table,
                                                                       window_size=window_size, negative_samples=0)
    positive_skip_grams_it = iter(positive_skip_grams)

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.

    # del _positive_skip_grams
    # del context
    # del label

    for _positive_skip_grams in positive_skip_grams_it:
        context_class = tf.expand_dims(tf.constant([_positive_skip_grams[1]], dtype="int64"), 1)
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=context_class,
                                                                                     num_true=1, num_sampled=num_ns,
                                                                                     unique=True,
                                                                                     range_max=vocab_size,
                                                                                     seed=seed,
                                                                                     name="negative_sampling")

        # Build context and label vectors (for one target word)
        negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
        # target = yield _positive_skip_grams[0]
        context = tf.concat([context_class, negative_sampling_candidates], 0)
        label = tf.constant([1] + [0] * num_ns, dtype="int64")
        # Append each element from the training example to global lists.
        targets_writer.writerow([_positive_skip_grams[0]])
        contexts_writer.writerow(list(np.array(context)[:, 0]))
        labels_writer.writerow(list(np.array(label)))
        del _positive_skip_grams
        del context
        del label
        del negative_sampling_candidates
        del context_class


def dataset_buffer_batch(targets, contexts, labels, buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def txt_to_pd(txt_path, sep="|||"):
    df = open(txt_path, "r", encoding='utf-8')
    lines = df.readlines()
    df.close()

    def split_line(index_line):
        if sep not in index_line[1] or len(index_line[1].strip().split(sep)) < 2:
            lines[index_line[0]] = (index_line[0] + 1, index_line[1].strip(), "")
        else:
            lines[index_line[0]] = (
                index_line[0] + 1, index_line[1].strip().split(sep)[0], index_line[1].strip().split(sep)[1])
        return lines

    list(map(split_line, enumerate(lines)))
    return lines


class GetNo(object):
    def __init__(self):
        self.key_i = 0

    def get_no(self):
        return self.key_i

    def add_no(self):
        self.key_i += 1
        if self.key_i % 10000 == 0:
            print(self.key_i)


def data_pre(text, stop_word):
    """
    分词，去停词，词性筛选
    :param text: 文本
    :param stop_word: 停用词
    :return:
    """
    GN = GetNo()
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
    seg = jieba.posseg.cut(text)

    def word_append(x):
        GN.add_no()
        if x.word not in stop_word and x.flag in pos:
            return x.word

    return list(filter(None, list(map(word_append, seg))))


def save_data_pre(data, stop_word, path="data/pre_data_pre.txt"):
    content_list = list(map(lambda x: x[1] + x[2], data))

    def key_str(x):
        key = " ".join(data_pre(x, stop_word)) + "\n"
        return key

    corpus = list(map(key_str, content_list))
    file = open(path, 'w', encoding='utf-8')

    file.writelines(corpus)
    file.close()


if __name__ == '__main__':
    pass
