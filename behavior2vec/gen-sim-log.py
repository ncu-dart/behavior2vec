#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script generates the simulation data
'''


import numpy as np
import os
import pickle
import sys


def normalize_list(li):
    return [float(e) / sum(li) for e in li]


def gen_items(n_items=100000, filename=''):
    '''
    Generation rule:
        1. Generate 100,000 items
        2. For each of the 100,000 items, generate x1 alternative items, x1 ~ max(1, round(N(mu=10, sigma=3))), assign alternative scores s1 to these x1 items, s1~unif(0.5, 1)
        3. For each of the 100,000 items, generate x2 affiliated items, x2 ~ max(1, round(N(mu=5, sigma=2))), assign affiliated scores s2 to these x2 items, s2~unif(0.5, 1)
    return:
        a dictionary of items of the following format
        items = {item-id: [(alter-item-list), (alter-item-prob-list)], [(affil-item-list, affil-item-prob-list), ...]]}
    '''
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
           items = pickle.load(f)
           return items


    items = {}
    for item_id in range(n_items):
        print("Generating item (%i / %i)" % (item_id + 1, n_items))
        n_alter_items = max(1, round(np.random.normal(loc=10, scale=3)))
        alter_items = np.random.choice(range(n_items), n_alter_items, replace=False).tolist()
        alter_scores = normalize_list(np.random.uniform(low=.5, high=1., size=n_alter_items).tolist())

        n_affil_items = max(1, round(np.random.normal(loc=5, scale=2)))
        affil_items = np.random.choice(range(n_items), n_affil_items, replace=False).tolist()
        affil_scores = normalize_list(np.random.uniform(low=.5, high=1., size=n_affil_items).tolist())
        items[item_id] = [[alter_items, alter_scores], [affil_items, affil_scores]]
    return items


def gen_next_item(cur_behavior, cur_item, session, items):
    prob = np.random.random()
    if prob < .1:
        return int(np.random.choice(session).split('-')[1])
    if cur_behavior == 'p':
        return np.random.choice(items[cur_item][1][0], p=items[cur_item][1][1]) if prob < .7 else np.random.choice(items[cur_item][0][0], p=items[cur_item][0][1])
    if cur_behavior == 'v':
        return np.random.choice(items[cur_item][0][0], p=items[cur_item][0][1]) if prob < .7 else np.random.choice(items[cur_item][1][0], p=items[cur_item][1][1])
    sys.exit(-1)


def gen_logs(items, n_sessions=10000, session_avg_len=10):
    '''
    Generation rule:
        1. generate 10000 users (sessions)
        2. The session length follows a exponential distribution, min session length = 2
        3. Each user randomly select one item as the inital item
        4. If a user continues to stay on the website, to view or to purchase is based on a uniform distribution
        5. If a user is viewing an item, randomly select the next item with prob. 0.1 and select the next item in alternative-item-list with prob. 0.6, and select the next item in affiliated-item-list with prob. 0.3
        6. If a user is purchasing an item and will view an item, randomly select the next item within the session with prob. .1 and select the next item in affiliated-item-list with prob. 0.6, and select the next item in alternative-item-list with prob. 0.3
    '''
    logs = []
    for uid in range(n_sessions):
        print("Generating session (%i / %i)" % (uid + 1, n_sessions))
        session_length = max(2, round(np.random.exponential(session_avg_len)))
        cur_behavior = 'v'
        cur_item = np.random.choice(list(items.keys()))
        session = ['%s-%i' % (cur_behavior, cur_item)]
        for i in range(session_length - 1):
            next_behavior = 'p' if np.random.random() < .1 else 'v'
            next_item = gen_next_item(cur_behavior, cur_item, session, items)
            session.append('%s-%i' % (next_behavior, next_item))
            cur_behavior, cur_item = next_behavior, next_item
        logs.append(session)
    return logs


def save_item_info(items, filename):
    if os.path.isfile(filename):
        return

    with open(filename, 'wb') as f:
        pickle.dump(items, f)


def save_log(logs):
    with open('./data/sim-log-session-%i.txt' % (len(logs)), 'w') as f:
        for session in logs:
            f.write('%s\n' % (' '.join(session)))


def main(argv):
    n_items = 10000
    n_sessions = 100000

    item_info_filename = './data/item-info-%i.pck' % (n_items)
    items = gen_items(n_items=n_items, filename=item_info_filename)
    save_item_info(items, filename=item_info_filename)
    logs = gen_logs(items, n_sessions=n_sessions)
    save_log(logs)


if __name__ == "__main__":
    main(sys.argv)


