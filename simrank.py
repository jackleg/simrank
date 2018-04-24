#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import itertools
import logging
import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

class BipartiteGraph(object):
    """bipartite graph를 표현하기 위한 클래스"""
    def __init__(self):
        self._lns = defaultdict(dict)
        self._rns = defaultdict(dict)

        # 노드 정보
        self._lns_node = set()
        self._rns_node = set()

    def add_edge(self, source, target, weight=1.0):
        """source ->(weight)-> target인 edge를 이 그래프에 추가 """

        # edge 추가 전에 node가 추가되어야 할 수도 있음.
        self.add_ln(source)
        self.add_rn(target)

        self._lns[source][target] = weight
        self._rns[target][source] = weight

    def add_ln(self, ln):
        self._lns_node.add(ln)

    def add_rn(self, rn):
        self._rns_node.add(rn)

    def has_ln(self, ln):
        return ln in self._lns_node

    def get_ln_edge_count(self, ln):
        return len(self._lns[ln])

    def get_rn_edge_count(self, rn):
        return len(self._rns[rn])

    def remove_edge(self, ln, rn):
        logging.info("remove edge: (%s, %s)" % (ln, rn))
        del self._lns[ln][rn]
        del self._rns[rn][ln]

    def remove_ln(self, ln):
        # self._lns[ln] 으로 바로 for를 돌리면, 아래의 RuntimeError 발생.
        # RuntimeError: dictionary changed size during iteration
        rns = list(self._lns[ln])

        for rn in rns:
            self.remove_edge(ln, rn)
            
            if len(self._rns[rn]) == 0:
                del self._rns[rn]
                self._rns_node.remove(rn)

        del self._lns[ln]
        self._lns_node.remove(ln)

    def get_lns(self):
        return self._lns_node

    def get_rns(self):
        return self._rns_node

    def get_lns_count(self):
        return len(self._lns_node)

    def get_rns_count(self):
        return len(self._rns_node)

    def get_weight(self, ln, rn):
        return self._lns[ln][rn]

    def get_lns_as_list(self):
        return list(self._lns_node)

    def get_rns_as_list(self):
        return list(self._rns_node)

    def get_lns_index(self):
        return dict([(node, i) for i, node in enumerate(self.get_lns_as_list())])

    def get_rns_index(self):
        return dict([(node, i) for i, node in enumerate(self.get_rns_as_list())])

    def get_ln_neighbors(self, ln):
        if ln not in self._lns:
            raise KeyError("%s is not in this graph's left side." % ln)

        return self._lns[ln]

    def get_rn_neighbors(self, rn):
        if rn not in self._rns:
            raise KeyError("%s is not in this graph's right side." % rn)

        return self._rns[rn]

    def get_neighbors(self, node, is_lns=True):
        if is_lns:
            return self.get_ln_neighbors(node)
        else:
            return self.get_rn_neighbors(node)

    def get_edge_count(self):
        count = 0
        for lns in self._lns:
            count += len(self._lns[lns])

        return count

    def split_subgraphs(self):
        """Bipartitle graph가 연결이 끊어진 여러 그래프로 나뉠 수 있다면, 해당 그래프들을 분리해서 list에 담아 반환한다."""

        logging.info("start to split graphs.")
        # not yet processed nodes
        unprocessed_lns = set(list(self.get_lns()))

        result_list = []
        while len(unprocessed_lns) > 0:
            g = BipartiteGraph()
            working_lns = [unprocessed_lns.pop()]

            while len(working_lns) > 0:
                ln = working_lns.pop(0)
                g.add_ln(ln)

                for rn in self.get_ln_neighbors(ln):
                    g.add_edge(ln, rn, self.get_weight(ln, rn))

                    # rn에 연결된 다른 ln 중 아직 처리되지 않은 node를 후보로 추가
                    for candidate_ln in self.get_rn_neighbors(rn):
                        if candidate_ln in unprocessed_lns:
                            unprocessed_lns.remove(candidate_ln)
                            working_lns.append(candidate_ln)

            # 하나의 subgraph 완성
            result_list.append(g)
            logging.info("    - complete new graph: %d-lns." % g.get_lns_count())

        return result_list

    def filter_edge(self, threshold=0.0):
        """weight가 threshold보다 작은 edge들을 제거한다.

        :return: 삭제된 edge들의 리스트
        """

        deleted_edges = []
        for ln in self.get_lns():
            ln_neighbors = list(self.get_ln_neighbors(ln).items())

            for rn, weight in ln_neighbors:
                if weight < threshold:
                    deleted_edges.append((ln, rn, weight))
                    self.remove_edge(ln, rn)

        return deleted_edges

    def print_graph(self):
        for ln in self.get_lns():
            try:
                for rn, weight in self.get_ln_neighbors(ln).iteritems():
                    sys.stdout.write("%s\t%s\t%f\n" % (ln, rn, self.get_weight(ln, rn)))
            # edge 없이 node만 있는 경우
            except KeyError:
                sys.stdout.write("%s\t\t\n" % ln)


def simrank_bipartite(G, r=0.8, max_iter=100, eps=1e-4):
    """ A bipartite version in the paper.
    """

    lns = G.get_lns()
    rns = G.get_rns()

    lns_count = len(lns)
    rns_count = len(rns)

    lns_index = G.get_lns_index()
    rns_index = G.get_rns_index()

    lns_sim_prev = np.identity(lns_count)
    lns_sim = np.identity(lns_count)

    rns_sim_prev = np.identity(rns_count)
    rns_sim = np.identity(rns_count)

    def _update_left_partite():
        for u, v in itertools.product(lns, lns):
            if u is v: continue

            u_index, v_index = lns_index[u], lns_index[v]
            u_ns, v_ns = G.get_ln_neighbors(u), G.get_ln_neighbors(v)

            if len(u_ns) == 0 or len(v_ns) == 0:
                lns_sim[u_index][v_index] = lns_sim[v_index][u_index] = 0.0

            else:
                # left의 neighbor들은 right
                s_uv = sum([rns_sim_prev[rns_index[u_n]][rns_index[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                lns_sim[u_index][v_index] = lns_sim[v_index][u_index] = (r * s_uv) / (len(u_ns) * len(v_ns))

    def _update_right_partite():
        for u, v in itertools.product(rns, rns):
            if u is v: continue

            u_index, v_index = rns_index[u], rns_index[v]
            u_ns, v_ns = G.get_rn_neighbors(u), G.get_rn_neighbors(v)

            if len(u_ns) == 0 or len(v_ns) == 0:
                rns_sim[u_index][v_index] = rns_sim[v_index][u_index] = 0.0

            else:
                # right의 neighbor들은 left
                s_uv = sum([lns_sim_prev[lns_index[u_n]][lns_index[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                rns_sim[u_index][v_index] = rns_sim[v_index][u_index] = (r * s_uv) / (len(u_ns) * len(v_ns))

    for i in range(max_iter):
        if np.allclose(lns_sim, lns_sim_prev, atol=eps) and np.allclose(rns_sim, rns_sim_prev, atol=eps):
            break

        lns_sim_prev = np.copy(lns_sim)
        rns_sim_prev = np.copy(rns_sim)

        _update_left_partite()
        _update_right_partite()

    print("Converge after %d iterations (eps=%f)." % (i, eps))

    return (lns_sim, rns_sim)


def simrank_double_plus_bipartite(G, r=0.8, max_iter=100, eps=1e-4):
    """ A simrank++ bipartite version in the paper.
    """

    lns = G.get_lns_as_list()
    rns = G.get_rns_as_list()

    lns_count = len(lns)
    rns_count = len(rns)

    lns_index = G.get_lns_index()
    rns_index = G.get_rns_index()

    lns_sim_prev = np.identity(lns_count)
    lns_sim = np.identity(lns_count)

    rns_sim_prev = np.identity(rns_count)
    rns_sim = np.identity(rns_count)

    # evidence
    lns_evidence = np.identity(lns_count)
    rns_evidence = np.identity(rns_count)

    # spread
    lns_spread = np.zeros(lns_count)
    rns_spread = np.zeros(rns_count)

    # normalized weight
    # row: node_from, column: node_to
    norm_weight_l_to_r = np.zeros((lns_count, rns_count))
    norm_weight_r_to_l = np.zeros((rns_count, lns_count))

    # transition_prob
    # row: node_from, column: node_to
    transition_prob_l_to_r = np.zeros((lns_count, rns_count))
    transition_prob_r_to_l = np.zeros((rns_count, lns_count))
    self_transition_prob_l = np.zeros(lns_count)
    self_transition_prob_r = np.zeros(rns_count)

    def _calculate_evidence(ns, ns_index, ns_evidence, is_lns=True):
        """ns의 evidence 계산하기

        ns의 evidence를 계산해서 ns_evidence에 저장한다.

        :param ns: evidence를 계산할 node list
        :param ns_index: ns의 배열에서의 index를 얻을 사전. key: node, value: index
        :param ns_evidence: 계산한 evidence를 저장할 matrix
        :param is_lns: ns가 left nodes인지 여부. True이면 left, False이면 right.
        """
        ## sum_i=1..n (1/2^n). 11개 이상은 1.0으로 봐도 됨.
        ## 공통 원소가 0개인 경우는 무조건 0이므로, 앞에 0 추가.
        calculated_evidence = [0.0,
                               0.50000, 0.75000, 0.87500, 0.93750, 0.96875,
                               0.98438, 0.99219, 0.99609, 0.99805, 0.99902]

        # u_index는 데이터가 저장되는 index가 아니라,
        # u x v pair를 만들 때 u 다음번 노드부터 사용하기 위해서만 사용한다.
        for u_index, u in enumerate(ns):
            for v in ns[(u_index+1):]:
                u_ns = set(G.get_neighbors(u, is_lns))
                v_ns = set(G.get_neighbors(v, is_lns))

                evidence_uv = 0.0

                if (len(u_ns) == 0) or (len(v_ns) == 0):
                    evidence_uv = 0.0
                else:
                    intersection = len(u_ns & v_ns)
                    evidence_uv = calculated_evidence[intersection] if intersection <= 10 else 1.0

                ns_evidence[ns_index[u]][ns_index[v]] = evidence_uv
                ns_evidence[ns_index[v]][ns_index[u]] = evidence_uv


    def _calculate_spread(ns, is_lns=True):
        """ns의 spread 계산"""
        for n in ns:
            nbr = G.get_neighbors(n, is_lns)
            weights = nbr.values()

            if is_lns:
                lns_spread[lns_index[n]] = np.exp(-np.var(weights))
            else:  # is_rns
                rns_spread[rns_index[n]] = np.exp(-np.var(weights))


    def _calculate_normalized_weight(ns, is_lns=True):
        for n in ns:
            nbrs = G.get_neighbors(n, is_lns)
            denom_factor = np.sum(nbrs.values())

            for nbr in nbrs:
                if is_lns:
                    norm_weight_l_to_r[lns_index[n]][rns_index[nbr]] = nbrs[nbr] / denom_factor
                else: # is_rns
                    norm_weight_r_to_l[rns_index[n]][lns_index[nbr]] = nbrs[nbr] / denom_factor


    def _calculate_transition_prob():
        logging.info("calculate spread for left side.")
        _calculate_spread(lns)
        logging.info("calculate spread for right side.")
        _calculate_spread(rns, False)

        logging.info("calculate normalized weight for left side.")
        _calculate_normalized_weight(lns)
        logging.info("calculate normalized weight for right side.")
        _calculate_normalized_weight(rns, False)

        # lns to rns
        logging.info("calculate lns to rns transition probability.")
        for n in lns:
            n_index = lns_index[n]

            sum_prob = 0.0
            for nbr in G.get_neighbors(n, True):
                nbr_index = rns_index[nbr]
                transition_prob_l_to_r[n_index][nbr_index] = rns_spread[nbr_index] * norm_weight_l_to_r[n_index][nbr_index]
                sum_prob += transition_prob_l_to_r[n_index][nbr_index]

            self_transition_prob_l[n_index] = 1.0 - sum_prob

        # rns to lns
        logging.info("calculate rns to lns transition probability.")
        for n in rns:
            n_index = rns_index[n]

            sum_prob = 0.0
            for nbr in G.get_neighbors(n, False):
                nbr_index = lns_index[nbr]
                transition_prob_r_to_l[n_index][nbr_index] = lns_spread[nbr_index] * norm_weight_r_to_l[n_index][nbr_index]
                sum_prob += transition_prob_r_to_l[n_index][nbr_index]

            self_transition_prob_r[n_index] = 1.0 - sum_prob


    def _update_left_partite():
        for u, v in itertools.product(lns, lns):
            if u is v: continue

            u_index, v_index = lns_index[u], lns_index[v]
            u_ns, v_ns = G.get_ln_neighbors(u), G.get_ln_neighbors(v)

            if len(u_ns) == 0 or len(v_ns) == 0:
                lns_sim[u_index][v_index] = lns_sim[v_index][u_index] = 0.0

            else:
                sim = 0.0

                for u_n, v_n in itertools.product(u_ns, v_ns):
                    # left의 neighbor들은 right
                    u_n_index = rns_index[u_n]
                    v_n_index = rns_index[v_n]

                    sim += transition_prob_l_to_r[u_index][u_n_index] * transition_prob_l_to_r[v_index][v_n_index] * rns_sim_prev[u_n_index][v_n_index]

                lns_sim[u_index][v_index] = lns_sim[v_index][u_index] = r * sim


    def _update_right_partite():
        for u, v in itertools.product(rns, rns):
            if u is v: continue

            u_index, v_index = rns_index[u], rns_index[v]
            u_ns, v_ns = G.get_rn_neighbors(u), G.get_rn_neighbors(v)

            if len(u_ns) == 0 or len(v_ns) == 0:
                rns_sim[u_index][v_index] = rns_sim[v_index][u_index] = 0.0

            else:
                sim = 0.0

                for u_n, v_n in itertools.product(u_ns, v_ns):
                    # right의 neighbor들은 left
                    u_n_index = lns_index[u_n]
                    v_n_index = lns_index[v_n]

                    sim += transition_prob_r_to_l[u_index][u_n_index] * transition_prob_r_to_l[v_index][v_n_index] * lns_sim_prev[u_n_index][v_n_index]

                rns_sim[u_index][v_index] = rns_sim[v_index][u_index] = r * sim


    ## evidance 계산
    logging.info("calculate evidence for left side.")
    _calculate_evidence(lns, lns_index, lns_evidence)
    logging.info("calculate evidence for right side.")
    _calculate_evidence(rns, rns_index, rns_evidence, False)

    ## transition probabiliyt
    logging.info("calculate transition probability.")
    _calculate_transition_prob()

    logging.debug("evidence")
    logging.debug(lns_evidence)
    logging.debug(rns_evidence)

    logging.debug("spread")
    logging.debug(lns_spread)
    logging.debug(rns_spread)

    logging.debug("norm weights")
    logging.debug(norm_weight_l_to_r)
    logging.debug(norm_weight_r_to_l)

    logging.debug("transition prob.")
    logging.debug(transition_prob_l_to_r)
    logging.debug(transition_prob_r_to_l)

    for i in range(max_iter):
        logging.info("start %d-iteration" % (i+1))
        _update_left_partite()
        _update_right_partite()

        logging.debug(lns_sim)
        logging.debug(rns_sim)

        if np.allclose(lns_sim, lns_sim_prev, atol=eps) and np.allclose(rns_sim, rns_sim_prev, atol=eps):
            break

        lns_sim_prev = np.copy(lns_sim)
        rns_sim_prev = np.copy(rns_sim)

    print("Converge after %d iterations (eps=%f)." % ((i+1), eps))

    return (np.multiply(lns_sim, lns_evidence),
            np.multiply(rns_sim, rns_evidence))


def convert_sim_to_dict(G, lns_sim, rns_sim, threshold=0.0):
    """bipartite graph G의 lns_sim, rns_sim 정보를 json object로 생성

    threshold보다 큰 값만 반환한다.
    """

    def _convert_sim_to_dict(ns, sim):
        result_dict = {}
        size = sim.shape[0]

        for outer_index in range(size):
            outer_node = ns[outer_index]
            result_dict[outer_node] = {}

            for inner_index in range(size):
                # 자기 자신은 제외
                if outer_index == inner_index: continue

                inner_node = ns[inner_index]

                if sim[outer_index][inner_index] > threshold:
                    result_dict[outer_node][inner_node] = sim[outer_index][inner_index]

        return result_dict

    lns_sim_dict = _convert_sim_to_dict(G.get_lns_as_list(), lns_sim)
    rns_sim_dict = _convert_sim_to_dict(G.get_rns_as_list(), rns_sim)

    return (lns_sim_dict, rns_sim_dict)


if __name__ == "__main__":
    print "example 1"
    print "camera --> hp.com, bestbuy.com"
    print "digital_camera --> hp.com, bestbuy.com"
    G = BipartiteGraph()

    G.add_edge("camera", "hp.com", 1.0)
    G.add_edge("camera", "bestbuy.com", 1.0)
    G.add_edge("digital_camera", "hp.com", 1.0)
    G.add_edge("digital_camera", "bestbuy.com", 1.0)

    lns_sim, rns_sim = simrank_double_plus_bipartite(G)

    print "sim"
    for node, index in G.get_lns_index().iteritems():
        print "%d | %s" % (index, node)
    print lns_sim

    for node, index in G.get_rns_index().iteritems():
        print "%d | %s" % (index, node)
    print rns_sim

    print "example2"
    print "pc --> hp.com"
    print "camera --> hp.com"
    G2 = BipartiteGraph()

    G2.add_edge("pc", "hp.com", 1.0)
    G2.add_edge("camera", "hp.com", 1.0)

    lns_sim, rns_sim = simrank_double_plus_bipartite(G2)

    print "sim"
    for node, index in G2.get_lns_index().iteritems():
        print "%d | %s" % (index, node)
    print lns_sim

    for node, index in G2.get_rns_index().iteritems():
        print "%d | %s" % (index, node)
    print rns_sim

    print "example3, ()안은 weight"
    print "순금반지 -> 1656770532(8), 1967201158(6), 1218341923(10), 886857215(5)"
    print "금반지24K -> 1218341923(6)"
    G3 = BipartiteGraph()

    G3.add_edge("순금반지", "1656770532", 8.0)
    G3.add_edge("순금반지", "1967201158", 6.0)
    G3.add_edge("순금반지", "1218341923", 10.0)
    G3.add_edge("순금반지", "886857215", 5.0)
    G3.add_edge("금반지24K", "1218341923", 6.0)

    lns_sim, rns_sim = simrank_double_plus_bipartite(G3)

    lns_sim_dict, rns_sim_dict = convert_sim_to_dict(G3, lns_sim, rns_sim)

    print "lns sim"
    print json.dumps(lns_sim_dict, sort_keys=True, indent=4, ensure_ascii=False)

    print "rns sim"
    print json.dumps(rns_sim_dict, sort_keys=True, indent=4, ensure_ascii=False)

    print "example4"
    print "split into subgraphs"
    print "A -> 1, 2 | B -> 1, 3 | C -> 3 | D -> 4, 5 | E -> 5"
    print "two subgraphs: (A, B, C) and (D, E)"

    G4 = BipartiteGraph()

    G4.add_edge("A", 1)
    G4.add_edge("A", 2)
    G4.add_edge("B", 1)
    G4.add_edge("B", 3)
    G4.add_edge("C", 3)
    G4.add_edge("D", 4)
    G4.add_edge("D", 5)
    G4.add_edge("E", 5)

    for c, subgraph in enumerate(G4.split_subgraphs(), start=1):
        print "%d-subgraph has %d-lns." % (c, subgraph.get_lns_count())
        print subgraph.get_lns()

    G5 = BipartiteGraph()

    G5.add_edge("A", 1)
    G5.add_edge("A", 2)
    G5.add_edge("B", 1)
    G5.add_edge("B", 3)
    G5.add_edge("C", 3)
    
    for c, subgraph in enumerate(G5.cut_subgraphs(), start=1):
        print "%d-subgraph has %d-lns." % (c, subgraph.get_lns_count())
        print subgraph.get_lns()

