import cv2
from matplotlib import pyplot as plt
import numpy as np


class Vertex:
    def __init__(self, key, data=None):
        self.data = data
        self.key = key

    def __eq__(self, other):
        if self.key == other.key:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.key)

    # def set_color(self, new_color):
    #     self.color = new_color
    #
    # def get_color(self):
    #     return self.color


class DataEdge:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def list_2_dict(v_list) -> dict:
    returned_dict = dict()
    for i in range(len(v_list)):
        returned_dict[v_list[i]] = i

    return returned_dict


class NeighborsList:
    def __init__(self, vertex_list=None):
        if vertex_list is None:
            vertex_list = []
        self.vertex_list = vertex_list

        self.vertex_dict = list_2_dict(self.vertex_list)
        self.neighborhood_list = [[] for x in range(len(self.vertex_dict))]

    def insertVertex(self, vertex):
        old_size = len(self.vertex_dict)
        self.vertex_dict[vertex] = len(self.vertex_dict)
        new_size = len(self.vertex_dict)

        if new_size > old_size:
            self.neighborhood_list.append([])

    def insertEdge(self, vertex1, vertex2, edge=1):
        vertex_idx1 = self.getVertexIdx(vertex1)
        vertex_idx2 = self.getVertexIdx(vertex2)

        self.neighborhood_list[vertex_idx1].append([vertex_idx2, edge])

    def deleteVertex(self, vertex):
        vertex_idx = self.getVertexIdx(vertex)

        del self.neighborhood_list[vertex_idx]
        for i in range(len(self.neighborhood_list)):
            idx_to_delete = []
            for j in range(len(self.neighborhood_list[i])):
                if self.neighborhood_list[i][j] == vertex_idx:
                    idx_to_delete.append(self.neighborhood_list[i][j])
                elif self.neighborhood_list[i][j] > vertex_idx:
                    self.neighborhood_list[i][j] -= 1
            for x in idx_to_delete:
                self.neighborhood_list[i].remove(x)

        # self.vertex_dict
        deleted_idx = self.vertex_dict[vertex]

        del self.vertex_dict[vertex]
        for i in range(deleted_idx, len(self.vertex_dict)):
            vertex_value_to_change = self.getVertex(i + 1)
            self.vertex_dict[vertex_value_to_change] = list(self.vertex_dict.items())[i][1] - 1

    def deleteEdge(self, vertex1, vertex2):
        vertex_idx1 = self.getVertexIdx(vertex1)
        vertex_idx2 = self.getVertexIdx(vertex2)

        x = self.neighborhood_list[vertex_idx1]
        for i in range(len(self.neighborhood_list[vertex_idx1])):
            t = self.neighborhood_list[vertex_idx1][i][0]
            if self.neighborhood_list[vertex_idx1][i][0] == vertex_idx2:
                del self.neighborhood_list[vertex_idx1][i]
                break

    def getVertexIdx(self, vertex):
        if vertex is None:
            return None
        else:
            return self.vertex_dict[vertex]

    def getVertex(self, vertex_id):
        if vertex_id is None:
            return None
        else:
            for key, value in list(self.vertex_dict.items()):
                if value == vertex_id:
                    return key
            return None

    def getVertexByKey(self, vertex_key):
        if vertex_key is None:
            return None
        else:
            for key, value in list(self.vertex_dict.items()):
                if key.key == vertex_key:
                    return key
            return None

    def order(self):
        return len(self.vertex_dict)

    def size(self):
        answer = 0
        for i in self.neighborhood_list:
            answer += len(i)
        return answer

    def edges(self):
        answer = []
        for i in range(len(self.neighborhood_list)):
            start = self.getVertex(i)
            for j in self.neighborhood_list[i]:
                end = self.getVertex(j)
                answer.append((start.key, end.key))
        return answer

    def neighbours(self, vertex_idx):
        lst = []
        for neighbour in self.neighborhood_list[vertex_idx]:
            lst.append([neighbour[0], neighbour[1]])
        return lst

    def __str__(self):
        answer = ""
        for i in range(len(self.neighborhood_list)):
            answer = answer + f"{i}: {self.neighborhood_list[i]} \n"
        return answer


def mst_prima(graph):
    size = graph.order()
    in_tree = [0] * size
    distance = [float('inf')] * size
    parent = [-1] * size

    basic_list = []
    # New graph without edges
    tree = NeighborsList()
    for i in list(graph.vertex_dict.items()):
        basic_list.append(i[0].key)
        tree.insertVertex(i[0])

    current = 0
    sum_of_wages = 0
    while in_tree[current] == 0:
        in_tree[current] = 1
        lst_of_neigh = graph.neighbours(current)
        for _, [idx, wage] in enumerate(lst_of_neigh):
            if distance[idx] > wage and in_tree[idx] == 0:
                distance[idx] = wage
                parent[idx] = current
        lowest_wage = float('inf')
        next_idx = 0
        for i, vertex in enumerate(basic_list):
            if in_tree[i] == 0:
                if distance[i] < lowest_wage:
                    lowest_wage = distance[i]
                    next_idx = i
        if next_idx != 0:
            tree.insertEdge(tree.getVertex(parent[next_idx]),
                            tree.getVertex(next_idx),
                            lowest_wage)
            tree.insertEdge(tree.getVertex(next_idx),
                            tree.getVertex(parent[next_idx]),
                            lowest_wage)
            sum_of_wages += lowest_wage
        current = next_idx
    print(in_tree)
    print(distance)
    print(parent)
    print("Tree length:", sum_of_wages)
    return tree


def bfs_fun(graph, start_vertex_idx, colour):
    stack = [start_vertex_idx]
    x = start_vertex_idx
    visited = []
    while stack:
        to_visit = stack.pop(0)
        visited.append(to_visit)
        x = graph.getVertex(to_visit).key
        graph.getVertex(to_visit).data = colour
        neighbours = graph.neighbours(to_visit)
        for neigh in neighbours:
            if neigh[0] not in visited:
                stack.append(neigh[0])


def printGraph(g):
    n = g.order()
    print("------GRAPH------", n)
    for i in range(n):
        v = g.getVertex(i).key
        print(v, end=" -> ")
        nbrs = g.neighbours(i)
        for (j, w) in nbrs:
            print(g.getVertex(j).key, w, end=";")
        print()
    print("-------------------")


def segmentation():
    image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
    image = image.astype('uint8')
    neighbors_matrix = NeighborsList()
    YY, XX = image.shape
    for i in range(YY):
        for j in range(XX):
            neighbors_matrix.insertVertex(Vertex(key=YY * j + i, data=image[i, j]))

    for i in range(1, YY - 1):
        for j in range(1, XX - 1):
            for n in range(3):
                for m in range(3):
                    diff = np.abs(image[i + n - 1, j + m - 1] - image[i, j])
                    neighbors_matrix.insertEdge(
                        neighbors_matrix.getVertex(YY * (j + m - 1) + i + n - 1),
                        neighbors_matrix.getVertex(YY * j + i),
                        diff)

    mst_graph = mst_prima(neighbors_matrix)

    longest_edge = -1
    longest_edge_idx = (-1, -1)
    for i in range(mst_graph.order()):
        curr_neighbours = mst_graph.neighbours(i)
        for j in curr_neighbours:
            if j[1] > longest_edge:
                longest_edge = j[1]
                longest_edge_idx = (i, j[0])
    mst_graph.deleteEdge(mst_graph.getVertex(longest_edge_idx[0]), mst_graph.getVertex(longest_edge_idx[1]))
    mst_graph.deleteEdge(mst_graph.getVertex(longest_edge_idx[1]), mst_graph.getVertex(longest_edge_idx[0]))

    IS = np.zeros((YY, XX), dtype='uint8')
    bfs_fun(mst_graph, longest_edge_idx[0], 50)
    bfs_fun(mst_graph, longest_edge_idx[1], 160)

    for i in range(1, YY - 1):
        for j in range(1, XX - 1):
            IS[i, j] = mst_graph.getVertex(YY * j + i).data
    plt.imshow(IS, 'gray')
    plt.show()


if __name__ == '__main__':
    segmentation()
