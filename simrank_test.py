#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import simrank

import numpy as np


G = simrank.BipartiteGraph()

for linecnt, line in enumerate(sys.stdin, start=1):
    tokens = line.rstrip("\n").split("\t")
    if (tokens[2] == "01") or (int(tokens[3]) < 5): continue

    G.add_edge(tokens[0], tokens[1], int(tokens[3]))

    if linecnt % 1000 == 0:
        sys.stderr.write("%d-processed.\n" % linecnt)


np.set_printoptions(threshold='nan')

for c, subgraph in enumerate(G.split_subgraphs(), start=1):
    print "%d-subgraph has %d-lns." % (c, subgraph.get_lns_count())
    
    query_sim, prod_sim = simrank.simrank_double_plus_bipartite(subgraph)

    for node, index in subgraph.get_lns_index().iteritems():
        print "%d | %s" % (index, node)

    for row_index, row in enumerate(query_sim):
        for col_index, val in enumerate(row):
            if val > 0.0: print row_index, col_index, val

    for node, index in subgraph.get_rns_index().iteritems():
        print "%d | %s" % (index, node)
    
    for row_index, row in enumerate(prod_sim):
        for col_index, val in enumerate(row):
            if val > 0.0: print row_index, col_index, val

