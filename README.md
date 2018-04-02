# 개요
* SimRank++: Query Rewriting through Link Analysis of the Click Graph 구현
* http://www.vldb.org/pvldb/1/1453903.pdf

# 제약
* bipartite graph 케이스에 대해서만 구현
* prerequisites: numpy

# 사용법
1. BipartiteGraph() 생성
```
G = BipartiteGraph()
```
2. BipartiteGraph.add_edge()로 edge를 추가해 그래프 완성
```
G.add_edge("camera", "hp.com", 1.0)
G.add_edge("camera", "bestbuy.com", 1.0)
G.add_edge("digital_camera", "hp.com", 1.0)
G.add_edge("digital_camera", "bestbuy.com", 1.0)
```
3. simrank_double_plus_bipartite(BipartiteGraph)로 유사도 계산. lns, rns 유사도를 따로 받음.
```
lns_sim, rns_sim = simrank_double_plus_bipartite(G)
```
