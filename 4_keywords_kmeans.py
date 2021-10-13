#!/usr/bin/python
# coding=utf-8
from common_util.data_util import get_word_list, txt_to_pd
from model_util.kmeans_model import get_word_list_score_sort
import pandas as pd
import tqdm


def main():
    data_content = "./data/data_sample.txt"
    sequences_path = "./data/cut_data_pre.txt"
    w2v_path = "./tmp_data/word2vectors.tsv"
    w2v_result_path = "./result/keys_tf_word2vec.csv"
    sequences_list = get_word_list(sequences_path)
    w2v = pd.read_csv(w2v_path)
    sequence_tqdm = tqdm.tqdm(iter(sequences_list))
    key_list = list(map(lambda x: get_word_list_score_sort(x, w2v), sequence_tqdm))
    data = txt_to_pd(data_content)
    id_list, job_list = list(map(lambda x: x[0], data)), list(map(lambda x: x[1], data))
    result = pd.DataFrame({"title": job_list, "key": key_list}, columns=['title', 'key'])
    result.to_csv(w2v_result_path, index=False)


if __name__ == '__main__':
    main()
