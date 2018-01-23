#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script shows the most similar behavior based on the behavior2vec model.

After installation, you may run the following command:

$ b2v-most-similar-behavior.py [test-file] [model-file] [output-file] [k] [behavior-type]

The [test-file] contains a list of query behaviors, one behavior per line.

The [output-file] is the output file, which contains a list of the top [k] most simiar behaviors
'''

import argparse
import pickle
import os
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
parser.add_argument("-b", "--behavior_type", type=str,
        help="Behavior type")
args = parser.parse_args()

import behavior2vec


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


def load_cur_behaviors(filename):
    cur_behaviors = []
    with open(filename) as f:
        for line in f:
            cur_behaviors.append(line.strip())
    return cur_behaviors


def save_result(filename, most_similar_behaviors):
    with open(filename, 'w') as f_out:
        for behavior_score_pairs in most_similar_behaviors:
            for i, (behavior, score) in enumerate(behavior_score_pairs):
                if i == 0:
                    f_out.write("%s:%f" % (behavior, score))
                else:
                    f_out.write(",%s:%f" % (behavior, score))
            f_out.write("\n")


def main(argv):
    check_args()

    most_similar_behaviors = []
    m = load_model(args.model_file)
    cur_behaviors = load_cur_behaviors(args.test_file)
    for b in cur_behaviors:
        sim_behaviors, sim_scores = m.most_similar_behavior(b, target_behavior_type=args.behavior_type, k=args.k)
        most_similar_behaviors.append([pair for pair in zip(sim_behaviors, sim_scores)])
    save_result(args.output_file, most_similar_behaviors)


if __name__ == "__main__":
    main(sys.argv)
