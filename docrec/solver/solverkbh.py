
# Algorithms-Book--Python
# Python implementations of the book "Algorithms - Dasgupta, Papadimitriou and Vazurani"
# https://github.com/israelst/Algorithms-Book--Python
parent = dict()
rank = dict()

def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0

def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]

def union(vertice1, vertice2, vertices):
    root1 = find(vertice1)
#    root2 = find(vertice2)
#    if root1 != root2:
    parent[vertice2] = root1
    for vertice in vertices:
        find(vertice)
        #else:
        #    parent[root1] = root2
        #    if rank[root1] == rank[root2]: rank[root2] += 1
        # if rank[root1] >= rank[root2]:
        #     parent[root2] = root1
        # else:
        #     parent[root1] = root2
        #     if rank[root1] == rank[root2]: rank[root2] += 1

def kruskal_based(graph):

    vertices = graph['vertices']
    for vertice in graph['vertices']:
        make_set(vertice)

    minimum_spanning_tree = set()
    forbidden_src = set()
    forbidden_dst = set()
    edges = list(graph['edges'])
    edges.sort()
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2): # not cycle
            if (vertice1 not in forbidden_src) and (vertice2 not in forbidden_dst): # path restriction
                union(vertice1, vertice2, vertices)
                minimum_spanning_tree.add(edge)
                forbidden_src.add(vertice1)
                forbidden_dst.add(vertice2)


    # for vertice in vertices:
    #     print(parent[vertice])
    N = len(vertices)
    solution = [parent[vertices[0]]]
    # print(minimum_spanning_tree)
    for i in range(N - 1):
        curr = solution[i]
        # print(i, curr)
        for _, u, v in minimum_spanning_tree:
            if u == curr:
                solution.append(v)
                break
    return solution

graph = {
        'vertices': ['A', 'B', 'C', 'D'],
        'edges': set([
            (1, 'A', 'B'),
            (5, 'A', 'C'),
            (3, 'A', 'D'),
            (4, 'B', 'C'),
            (2, 'B', 'D'),
            (1, 'C', 'D'),
            ])
        }

graph = {
        'vertices': ['A', 'B', 'C', 'D'],
        'edges': set([
            (1, 'A', 'B'),
            (5, 'A', 'C'),
            (3, 'A', 'D'),
            (4, 'B', 'C'),
            (2, 'B', 'D'),
            (1, 'C', 'D'),
            ])
        }
# minimum_spanning_tree = set([
#             (1, 'A', 'B'),
#             (2, 'B', 'D'),
#             (1, 'C', 'D'),
#             ])
# assert kruskal(graph) == minimum_spanning_tree