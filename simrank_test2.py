#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import simrank
import json

import numpy as np
from tqdm import tqdm

G = simrank.BipartiteGraph()

for line in tqdm(sys.stdin):
    tokens = line.rstrip("\n").split("\t")
    if float(tokens[2]) < 5: continue

    G.add_edge(tokens[0], tokens[1], float(tokens[2]))


np.set_printoptions(threshold='nan')


result_subgraphs = filter(lambda g: (g.get_lns_count() > 1) or (g.get_rns_count() > 1), G.split_subgraphs())
threshold = 6.0
while any([g.get_lns_count() > 1000 for g in result_subgraphs]):
    print "edge threshold: 6.0"
    new_subgraphs = []
    for index, s in enumerate(result_subgraphs):
        if s.get_lns_count() > 1000:
            result_subgraphs.remove(s)

            s.filter_edge(threshold)

            new_subgraphs.extend(filter(lambda g: (g.get_lns_count() > 1) or (g.get_rns_count() > 1), s.split_subgraphs()))
        else:
            new_subgraphs.append(s)

    result_subgraphs = new_subgraphs
    threshold += 1.0

          
for c, subgraph in enumerate(result_subgraphs, start=1):
    print "%d-subgraph" % c
    print "    has %d-lns / %d-rns" % (subgraph.get_lns_count(), subgraph.get_rns_count())

    print "========= graph"
    subgraph.print_graph()

    query_sim, prod_sim = simrank.simrank_double_plus_bipartite(subgraph)
    query_sim_dict, prod_sim_dict = simrank.convert_sim_to_dict(subgraph, query_sim, prod_sim)

    print "========= lns"
    for node, index in subgraph.get_lns_index().iteritems():
        print "%d | %s" % (index, node)

    for row_index, row in enumerate(query_sim):
        for col_index, val in enumerate(row):
            if val > 0.0: print row_index, col_index, val

    print json.dumps(query_sim_dict, sort_keys=True, indent=4, ensure_ascii=False)

    print "========= rns"
    for node, index in subgraph.get_rns_index().iteritems():
        print "%d | %s" % (index, node)
    
    for row_index, row in enumerate(prod_sim):
        for col_index, val in enumerate(row):
            if val > 0.0: print row_index, col_index, val

    print json.dumps(prod_sim_dict, sort_keys=True, indent=4, ensure_ascii=False)
