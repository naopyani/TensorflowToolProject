#!/usr/bin/python
# coding=utf-8
from common_util.data_util import text_line_object
from common_util.data_util import get_sequences_list
import io


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    data_path = "./data/cut_data_pre.txt"
    sequences_path = "./tmp_data/sequences.tsv"
    vocab_path = "./tmp_data/vocab.tsv"
    vocab_size = 193315
    sequence_length = None
    text_ds = text_line_object(data_path)
    sequences, vocab = get_sequences_list(text_ds, vocab_size, sequence_length=sequence_length, batch_size=2)
    out_sequences = io.open(sequences_path, 'w', encoding='utf-8')
    for _sequences in sequences:
        out_sequences.write('\t'.join([str(x) for x in list(_sequences)]) + "\n")
    out_sequences.close()

    out_vocab = io.open(vocab_path, 'w', encoding='utf-8')
    for index, word in enumerate(vocab):
        out_vocab.write(word + "\n")
    out_vocab.close()


if __name__ == '__main__':
    main()
