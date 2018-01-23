#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script shows the most similar items based on the behavior2vec model.

After installation, you may run the following command:

$ b2v-most-similar-item.py [test-file] [model-file] [output-file] [k]

The [test-file] contains a list of query items, one behavior per line.
The [output-file] is the output file, which contains a list of the top [k] most simiar items
'''

import argparse
import os
import pickle
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-te", "--test_file", type=str,
        help="Test file")
parser.add_argument("-m", "--model_file", type=str,
        help="Model file")
parser.add_argument("-o", "--output_file", type=str,
        help="Output file")
parser.add_argument("-k", "--k", type=int, default=1,
        help="Number of the most similar behaviors to return")
args = parser.parse_args()


def check_args():
    if args.test_file is None or not os.path.isfile(args.test_file):
        print("test_file argument must be a valid file")
        parser.print_help()
        sys.exit(-1)

    if args.model_file is None or not os.path.isfile(args.model_file):
        print("model_file argument must be a valid file")
        parser.print_help()
        sys.exit(-1)


def load_model(filename):
    with open(filename, 'rb') as f:
        m = pickle.load(f)
    return m


def load_cur_items(filename):
    cur_items = []
    with open(filename) as f:
        for line in f:
            cur_items.append(line.strip())
    return cur_items


def save_result(filename, most_similar_items):
    with open(filename, 'w') as f_out:
        for item_score_pairs in most_similar_items:
            for i, (item, score) in enumerate(item_score_pairs):
                if i == 0:
                    f_out.write("%s:%f" % (item, score))
                else:
                    f_out.write(",%s:%f" % (item, score))
            f_out.write("\n")


def main(argv):
    check_args()

    most_similar_items = []
    m = load_model(args.model_file)
    cur_items = load_cur_items(args.test_file)
    for item in cur_items:
        sim_items, sim_scores = m.most_similar_item(item, k=args.k)
        most_similar_items.append([pair for pair in zip(sim_items, sim_scores)])
    save_result(args.output_file, most_similar_items)


if __name__ == "__main__":
    main(sys.argv)
