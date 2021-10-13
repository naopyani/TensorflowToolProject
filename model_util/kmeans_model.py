#!/usr/bin/python
# coding=utf-8

from sklearn.cluster import KMeans
import numpy as np


def get_word_list_score_sort(sequence_list, w2v, top_k=10):
    """
    给出句子序列sequence列表的k-means的欧式距离得分，返回topK关键词
    :param top_k: 关键词数量
    :param sequence_list:词语列表
    :param w2v:词向量
    :return:
    """
    simple_array = np.array(list(map(lambda x: w2v[w2v["word"] == x].values[0][1:], sequence_list)))
    kmeans = KMeans(n_clusters=1, random_state=10).fit(simple_array)
    vec_center = kmeans.cluster_centers_
    vec_center = vec_center[0]  # 第一个类别聚类中心,本例只有一个类别
    # 计算距离（相似性） 采用欧几里得距离（欧式距离）
    distances = list(map(lambda x: np.sqrt(np.sum(np.square(x - vec_center))), simple_array))
    # 去重、按距离升序排列、去除分值保留词、取topK
    zip_keyword_score = list(set(zip(sequence_list, distances)))
    keyword_list = " ".join(list(map(lambda x: x[0], sorted(zip_keyword_score, key=lambda x: x[1])[:top_k])))
    return keyword_list


if __name__ == '__main__':
    pass
