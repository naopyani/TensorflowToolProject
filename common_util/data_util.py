#!/usr/bin/python
# coding=utf-8
import os
from urllib.request import urlretrieve
import zipfile
import collections
import tensorflow as tf
import tqdm
import re
import string
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


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
    df = open(word_pre_path, "r", encoding='utf-8')
    corpus = df.readlines()
    df.close()
    raw_word = list(map(lambda y: list(str(y).replace("\n", "").split(" ")), corpus))
    return raw_word


def get_word_bag(word_pre_path):
    word_bag = list()
    df = open(word_pre_path, "r", encoding='utf-8')
    corpus = df.readlines()
    df.close()
    list(map(lambda y: word_bag.extend(list(str(y).replace("\n", "").split(" "))), corpus))
    return word_bag


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    word2id = dict()
    for word, _ in count:
        word2id[word] = len(word2id)
    data = list()
    unk_count = 0
    for word in words:
        if word in word2id:
            index = word2id[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return data, count, word2id, id2word


def custom_standardization(input_data):
    """
    把传入的张量或字符串变成小写，和去掉标点符号
    :param input_data: 张量或者字符串
    :return:
    """
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                    '[%s]' % re.escape(string.punctuation), '')


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
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)
    # inverse_vocab = vectorize_layer.get_vocabulary()
    vectorize_layer.adapt(text_ds.batch(batch_size))
    # text向量化
    text_vector_ds = text_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())
    return sequences, vectorize_layer


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    """
    生成训练数据
    :param sequences:
    :param window_size:
    :param num_ns:
    :param vocab_size:
    :param seed:
    :return:
    """
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def dataset_buffer_batch(targets, contexts, labels, buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


if __name__ == '__main__':
    pass
    # raw_word = get_word_bag("../data/XXX.txt")
    # data, count, vocab, inverse_vocab = build_dataset(raw_word)
    # print(data, count, vocab, inverse_vocab)
