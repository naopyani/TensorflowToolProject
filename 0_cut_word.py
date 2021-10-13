#!/usr/bin/python
# coding=utf-8

from common_util.data_util import txt_to_pd, save_data_pre
import codecs


def main():
    data_content = "./data/data_sample.txt"
    sequences_path = "./data/cut_data_pre.txt"
    stop_word_path = './data/stop_word.txt'
    data = txt_to_pd(data_content)
    stop_word = list(map(lambda x: x.strip(), codecs.open(stop_word_path, 'r', encoding='utf-8').readlines()))
    save_data_pre(data, stop_word, sequences_path)


if __name__ == '__main__':
    main()
