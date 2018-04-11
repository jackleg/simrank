# 개요
* SimRank++: Query Rewriting through Link Analysis of the Click Graph 구현
    * bipartite 케이스에 대해서만 구현
* http://www.vldb.org/pvldb/1/1453903.pdf

# prerequisites
* numpy

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

3. (Optional) sub graph로 split하기
- 필요하다면 sub graph로 graph를 분리해서 계산 시간이나 메모리 사용량을 줄일 수 있다.
```
subgraph_list = G.split_subgraphs()
```

4. simrank_double_plus_bipartite(BipartiteGraph)로 유사도 계산. lns, rns 유사도를 따로 받음.
```
lns_sim, rns_sim = simrank_double_plus_bipartite(G)
```

# Helper
* Similarty를 노드 실제 값들로 출력할 수 있도록 사전(dict)형으로 정리해주는 convert_sim_to_dict
    * 아래와 같이 graph와 계산한 lns_sim, rns_sim matrix를 넘겨주면, 각각 사전형을 반환
```
...
> lns_sim_dict, rns_sim_dict = convert_sim_to_dict(G3, lns_sim, rns_sim)
> print json.dumps(lns_sim_dict, sort_keys=True, indent=4, ensure_ascii=False)
{
    "금반지24K": {
        "순금반지": 9.405719204286108e-05
    },
    "순금반지": {
        "금반지24K": 9.405719204286108e-05
    }
}
```

# 결과 예시
```
> ./simrank.py
...
example 1
camera --> hp.com, bestbuy.com
digital_camera --> hp.com, bestbuy.com
Converge after 10 iterations (eps=0.000100).
sim
0 | camera
1 | digital_camera
[[ 1.          0.49994757]
 [ 0.49994757  1.        ]]
0 | bestbuy.com
1 | hp.com
[[ 1.          0.49994757]
 [ 0.49994757  1.        ]]
example2
pc --> hp.com
camera --> hp.com
Converge after 2 iterations (eps=0.000100).
sim
0 | pc
1 | camera
[[ 1.   0.4]
 [ 0.4  1. ]]
0 | hp.com
[[ 1.]]
example3, ()안은 weight
순금반지 -> 1656770532(8), 1967201158(6), 1218341923(10), 886857215(5)
금반지24K -> 1218341923(6)
Converge after 2 iterations (eps=0.000100).
sim
0 | 순금반지
1 | 금반지24K
[[  1.00000000e+00   4.77748390e-05]
 [  4.77748390e-05   1.00000000e+00]]
0 | 1656770532
2 | 1967201158
1 | 1218341923
3 | 886857215
[[  1.00000000e+00   1.57029184e-04   2.50690679e-04   2.50690679e-04]
 [  1.57029184e-04   1.00000000e+00   1.57029184e-04   1.57029184e-04]
 [  2.50690679e-04   1.57029184e-04   1.00000000e+00   2.50690679e-04]
 [  2.50690679e-04   1.57029184e-04   2.50690679e-04   1.00000000e+00]]
example4
split into subgraphs
A -> 1, 2 | B -> 1, 3 | C -> 3 | D -> 4, 5 | E -> 5
two subgraphs: (A, B, C) and (D, E)
1-subgraph has 3-lns.
['A', 'C', 'B']
2-subgraph has 2-lns.
['E', 'D']
```
