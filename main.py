import numpy as np
import math


# idk why i made this but here it is
class Heap:

    def __init__(self, array=None):
        if array is not None:
            self.heap = array
            self.build_min_heap()
        else:
            self.heap = []

    def print_heap(self):
        print(self.heap)

    @staticmethod
    def parent_index(index):
        return math.floor((index - 1)/2)

    def parent(self, index):
        if index == 0:
            return self.heap[0]
        else:
            return self.heap[self.parent_index(index)]

    @staticmethod
    def left(index):
        return (index * 2) + 1

    def left_child(self, index):
        index = self.left(index)
        if index > len(self.heap):
            return None
        else:
            return self.heap[index]

    @staticmethod
    def right(index):
        return (index + 1) * 2

    def right_child(self, index):
        index = self.right(index)
        if index > len(self.heap):
            return None
        else:
            return self.heap[index]

    def perc_down_min(self, index):
        if self.left(index) >= len(self.heap) or self.left_child(index) is not None:
            if self.right(index) >= len(self.heap) or self.heap[self.right(index)] is None:
                child_index = self.left(index)
            elif self.right_child(index) < self.left_child(index):
                child_index = self.right(index)
            else:
                child_index = self.left(index)
            if child_index < len(self.heap):
                if self.heap[child_index] < self.heap[index]:
                    self.heap[index], self.heap[child_index] = self.heap[child_index], self.heap[index]
                    self.perc_down_min(child_index)

    def build_min_heap(self):
        start = int(len(self.heap) / 2) - 1
        for index in range(start, -1, -1):
            self.perc_down_min(index)

    def change_key(self, index, key):
        if index < len(self.heap):
            self.heap[index] = key
            while index > 0 and self.parent(index) > self.heap[index]:
                self.heap[index], self.heap[self.parent_index(index)] = self.parent(index), self.heap[index]
                index = self.parent_index(index)
            self.perc_down_min(index)

    def extract_min(self):
        min_element = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self.perc_down_min(0)
        return min_element

    def insert_element(self, key):
        self.heap.append(key)
        index = len(self.heap) - 1
        while index > 0 and self.parent(index) > self.heap[index]:
            self.heap[index], self.heap[self.parent_index(index)] = self.parent(index), self.heap[index]
            index = self.parent_index(index)

    def is_empty(self):
        return len(self.heap) == 0

    def is_not_empty(self):
        return len(self.heap) != 0

    def in_heap(self, number):
        return number in self.heap



class VertexHeap(Heap):

    def __init__(self, vertexes_edges):
        super(VertexHeap, self).__init__(vertexes_edges)

    def perc_down_min(self, index):
        if self.left(index) >= len(self.heap) or self.left_child(index) is not None:
            if self.right(index) >= len(self.heap) or self.heap[self.right(index)] is None:
                child_index = self.left(index)
            elif self.right_child(index).get_distance() < self.left_child(index).get_distance():
                child_index = self.right(index)
            else:
                child_index = self.left(index)
            if child_index < len(self.heap):
                if self.heap[child_index].get_distance() < self.heap[index].get_distance():
                    self.heap[index], self.heap[child_index] = self.heap[child_index], self.heap[index]
                    self.perc_down_min(child_index)

    def in_heap(self, vertex):
        return vertex in self.heap

class Link:

    def __init__(self, value, head_link=None, tail_link=None):
        self.value = value
        self.head_link = head_link
        self.tail_link = tail_link

    def set_head_link(self, link):
        self.head_link = link

    def set_tail_link(self, link):
        self.tail_link = link

    def get_value(self):
        return self.value

    def get_tail_link(self):
        return self.tail_link

    def get_head_link(self):
        return self.head_link


class LinkedList:

    def __init__(self, head=None, tail=None):
        self.head = head
        self.tail = tail
        self.length = 0

    def set_head_link(self, link):
        self.head = link

    def set_tail_link(self, link):
        self.tail = link

    def get_tail_link(self):
        return self.tail

    def get_head_link(self):
        return self.head

    def insert_at_head(self, value):
        self.head = Link(value, self.head)
        self.length += 1
        if self.length == 1:
            self.tail = self.head

    def get_length(self):
        return self.length

    def pop_head(self):
        temp = self.head
        self.head = self.head.get_tail_link()
        self.length -= 1
        if self.length == 0:
            self.tail = None
        return temp

    def get_item_at_pos(self, pos):
        temp = self.head
        if pos < self.length:
            for i in range(pos):
                temp = temp.get_tail_link()
            return temp.get_value()

    def inset_item_at_pos(self, value, pos):
        self.length += 1
        temp = self.head
        if pos < self.length:
            for i in range(pos - 1):
                temp = temp.get_tail_link()
            new_link = Link(value, temp.get_head_link, temp)
            temp.set_head_link(new_link)
            temp.get_head_link().set_tail_link(new_link)
            if pos == self.length:
                self.tail = new_link

    def print_linked_list(self):
        temp = self.head
        for i in range(self.length):
            print(temp.get_value())
            temp = temp.get_tail_link()


class DisjointSet:

    def __init__(self, num_sets=0):
        self.num_sets = num_sets
        self.parents = list(range(num_sets))
        self.ranks = [0] * num_sets
        self.sizes = [1] * num_sets

    def make_set(self):
        self.num_sets += 1
        self.parents.append(len(self.parents))
        self.ranks.append(0)
        self.sizes.append(1)

    def find(self, index):
        if 0 <= index < len(self.parents):
            pass
        parent = self.get_parent(index)
        if parent == index:
            return index
        while True:
            new_parent = self.get_parent(parent)
            if new_parent == parent:
                return parent
            parent = self.get_parent(parent)

    def in_same_set(self, index1, index2):
        if self.find(index1) == self.find(index2):
            return True
        return False

    def get_size_of_set(self, index):
        return self.sizes[self.find(index)]

    def union(self, index1, index2):
        set1 = self.find(index1)
        set2 = self.find(index2)
        if set1 == set2:
            return False
        if self.ranks[set1] == self.ranks[set2]:
            self.ranks[set1] += 1
        elif self.ranks[set1] < self.ranks[set2]:
            self.ranks[set1], self.ranks[set2] = self.ranks[set2], self.ranks[set1]

        self.parents[set2] = set1
        self.sizes[set1] += self.sizes[set2]
        self.sizes[set2] = 0
        self.num_sets -= 1
        return True

    def get_parent(self, index):
        return self.parents[index]

    def get_num_sets(self):
        return self.num_sets

    def print(self):
        print("parents")
        print(self.parents)
        print('ranks')
        print(self.ranks)
        print('sizes')
        print(self.sizes)


class DisjointSetVert(DisjointSet):

    def __init__(self, vertex_list=None):
        if vertex_list is not None:
            num_sets = len(vertex_list) - 1
        else:
            num_sets = 0
        super().__init__(num_sets)
        if vertex_list is not None:
            self.parents = vertex_list
        else:
            self.parents = []

    def make_set_vert(self, vert):
        self.num_sets += 1
        self.parents.append(vert)
        self.ranks.append(0)
        self.sizes.append(1)

    def find(self, index):
        if 0 <= index < len(self.parents):
            pass
        parent = self.get_parent_index(index)
        if parent == index:
            return index
        while True:
            new_parent = self.get_parent_index(index)
            if new_parent == parent:
                return parent
            parent = self.get_parent_index(index)

    def get_parent_index(self, index):
        return self.parents[index].get_vertex_index()

    def union(self, index1, index2):
        set1 = self.find(index1)
        set2 = self.find(index2)
        if set1 == set2:
            return False
        if self.ranks[set1] == self.ranks[set2]:
            self.ranks[set1] += 1
        elif self.ranks[set1] < self.ranks[set2]:
            self.ranks[set1], self.ranks[set2] = self.ranks[set2], self.ranks[set1]

        self.parents[set2] = self.get_parent(set1)
        self.sizes[set1] += self.sizes[set2]
        self.sizes[set2] = 0
        self.num_sets -= 1
        return True

    def print(self):
        super(DisjointSetVert, self).print()


class Queue:

    def __init__(self):
        self.q = []
        self.end_q = 0

    def enqueue(self, o):
        self.q.append(o)
        self.end_q += 1

    def dequeue(self):
        self.end_q -= 1
        return self.q.pop(0)

    def peek(self):
        return self.q[0]

    def len_queue(self):
        return len(self.q)

    def is_empty(self):
        if len(self.q) == 0:
            return True
        return False

    def not_empty(self):
        if self.is_empty():
            return False
        return True

    def print_queue(self):
        print(self.q)


class Vertex:

    def __init__(self, vertex_index, vertex_signifier=None):
        self.vertex_index = vertex_index
        self.vertex_signifier = vertex_signifier
        self.prior = None
        self.color = None
        self.distance = 0
        self.end_time = 0

    def __repr__(self):
        return str('Vertex at pos ' + str(self.get_vertex_index()))

    def __str__(self):
        return str('Vertex at pos ' + str(self.get_vertex_index()))

    def set_breath_search(self, color, distance, prior):
        self.set_color(color)
        self.distance = distance
        self.prior = prior

    def set_depth_search(self, color, end_time, prior):
        self.color = color
        self.end_time = end_time
        self.prior = prior

    def set_vertex_end_time(self, time):
        self.end_time = time

    def get_end_time(self):
        return self.end_time

    def set_vertex_index(self, vertex_index):
        self.vertex_index = vertex_index

    def get_vertex_index(self):
        return self.vertex_index

    def set_vertex_signifier(self, vertex_signifier):
        self.vertex_signifier = vertex_signifier

    def get_vertex_signifier(self):
        return self.vertex_signifier

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color

    def set_prior(self, prior):
        self.prior = prior

    def get_prior(self):
        return self.prior

    def set_distance(self, distance):
        self.distance = distance

    def get_distance(self):
        return self.distance


class AdjacentList:

    def __init__(self):
        self.adj_list = []

    def new_list(self):
        self.adj_list.append([])

    def new_edge(self, index, vertex, weight):
        self.adj_list[index].append([vertex, weight])

    def gather_edges_in_list(self):
        temp_list = []
        for ind in range(len(self.adj_list)):
            for inner in self.adj_list[ind]:
                inner_list = [ind, inner[0], inner[1]]
                temp_list.append(inner_list)
        return temp_list

    def get_adjacent_vertex(self, index):
        vertex_list = []
        for vert in self.adj_list[index]:
            vertex_list.append(vert[0])
        return vertex_list

    def get_adjacent_vertex_edges(self, index):
        return self.adj_list[index]

    def print(self):
        i = 0
        for ind in self.adj_list:
            print(f'STARTING WITH INDEX {i}')
            for inner in ind:
                print(f'Vertex index {inner[0].get_vertex_index()}')
                print(f'Vertex signify {inner[0].get_vertex_signifier()}')
                print(f'Edge weight {inner[1]}\n')
            i += 1


class Graph:

    def __init__(self, file_path, signifier_index=None):
        self.adj_matrix = self.adjacent_matrix_from_file(file_path, signifier_index)
        self.adj_list = AdjacentList()
        self.vertexes = []
        self.adjacent_list_from_file(file_path, signifier_index)
        self._time = 0

    def get_num_vertexes(self):
        return len(self.vertexes)

    def reset_time(self):
        self._time = 0

    def time_plus_one(self):
        self._time += 1
        return self._time

    def get_time(self):
        return self._time

    def set_time_zero(self):
        self._time = 0

    def adjacent_matrix_from_file(self, file_path, signify_index=None):
        with open(file_path, 'r+') as file:
            adj = []
            for line in file.readlines():
                line_list = line.strip('\n').split(',')
                if signify_index is not None:
                    line_list.pop(signify_index)
                    adj.append(line_list)
                else:
                    adj.append(line_list)
            self.adj_matrix = np.array(adj)
            return self.adj_matrix

    def adjacent_list_from_file(self, file_path, signify_index=None):
        n = 0
        with open(file_path, 'r+') as file:
            single = file.readline().strip('\n').split(',')
            if signify_index is not None:
                single.pop(signify_index)
            n = len(single)
            self.vertexes = [None] * n
            for i in range(n):
                self.vertexes[i] = Vertex(i)
        with open(file_path, 'r+') as file:
            i = 0
            for line in file.readlines():
                line_list = line.strip('\n').split(',')
                self.adj_list.new_list()
                if signify_index is not None:
                    self.vertexes[i].set_vertex_signifier(line_list.pop(signify_index))
                for k in range(n):
                    if int(line_list[k]) != 0:
                        self.adj_list.new_edge(i, self.vertexes[k], int(line_list[k]))
                i += 1
        return self.adj_list

    def print_adj_list(self):
        self.adj_list.print()

    def print_adj_matrix(self):
        print(self.adj_matrix)

    def get_vertex_with_sig(self, signifier):
        for vert in self.vertexes:
            if vert.get_vertex_signifier() == signifier:
                return vert
        return None

    def get_adjacent_vert(self, index):
        return self.adj_list.get_adjacent_vertex(index)

    def breadth_first_search(self, signifier):
        take_vert = None
        for vert in self.vertexes:
            if vert.get_vertex_signifier() != signifier:
                vert.set_color('white')
                vert.set_distance = math.inf
                vert.set_prior = None
            else:
                take_vert = vert
                take_vert.set_color('gray')
                take_vert.set_distance(0)
                take_vert.set_prior(None)
        queue = Queue()
        queue.enqueue(take_vert)
        while queue.not_empty():
            vert_origin = queue.dequeue()
            list_vert = self.get_adjacent_vert(vert_origin.get_vertex_index())
            for vert in list_vert:
                if vert.get_color() == 'white':
                    vert.set_breath_search('gray', vert_origin.get_distance() + 1, vert_origin)
                    queue.enqueue(vert)
            vert_origin.set_color('black')

    def print_vert_bfs(self):
        for vertex in self.vertexes:
            sig = vertex.get_vertex_signifier()
            color = vertex.get_color()
            distance = vertex.get_distance()
            prior = vertex.get_prior()
            if prior is None:
                prior = 'None'
            else:
                prior = prior.get_vertex_signifier()
            print(f'Vertex sig {sig}.\n'
                  f'    Vertex color {color}, distance {distance},'
                  f' prior {prior}')

    def simple_vert_print(self):
        for vert in self.vertexes:
            print(vert.get_vertex_signifier())

    def topographical_sort(self):
        linked_list = LinkedList()
        self.depth_first_search(linked_list)
        return linked_list

    def depth_first_search(self, linked_list=None):
        for vert in self.vertexes:
            vert.set_color('white')
            vert.set_prior(None)
        self.set_time_zero()
        for vert in self.vertexes:
            if vert.get_color() == 'white':
                if linked_list is None:
                    self.depth_first_visit(vert, linked_list)
                else:
                    self.depth_first_search(vert)

    def depth_first_visit(self, vertex, linked_list=None):
        vertex.set_distance(self.time_plus_one())
        vertex.set_color('gray')
        for vert in self.get_adjacent_vert(vertex.get_vertex_index()):
            if vert.get_color() == 'white':
                vert.set_prior(vertex)
                self.depth_first_visit(vert)
        vertex.set_color('black')
        vertex.set_vertex_end_time(self.time_plus_one())
        if linked_list is not None:
            linked_list.insert_at_head(vertex)

    def print_vert_dfs(self):
        for vertex in self.vertexes:
            sig = vertex.get_vertex_signifier()
            color = vertex.get_color()
            distance = vertex.get_distance()
            prior = vertex.get_prior()
            time = vertex.get_end_time()
            if prior is None:
                prior = 'None'
            else:
                prior = prior.get_vertex_signifier()
            print(f'Vertex sig {sig}.\n'
                  f'    Vertex color {color},'
                  f' prior {prior}, start_time {distance}, finish_time {time}')

    def init_single_source(self, vertex_signifier):
        for vert in self.vertexes:
            vert.set_distance(math.inf)
            vert.set_prior(None)
        vert = self.get_vertex_with_sig(vertex_signifier)
        vert.set_distance(0)

    @staticmethod
    def relax(from_vector, to_vector, weight):
        if to_vector.get_distance() > from_vector.get_distance() + weight:
            to_vector = from_vector + weight
            to_vector.set_prior(from_vector)

    def gather_edges_in_list(self):
        edges = self.adj_list.gather_edges_in_list()
        for edge in edges:
            for vert in self.vertexes:
                if edge[0] == vert.get_vertex_index():
                    edge[0] = vert
                    break
        return edges

    def insertion_sort_edges(self):
        edges = self.gather_edges_in_list()
        for i in range(1, len(edges)):
            key = edges[i]
            k = i - 1
            while k >= 0 and key[2] < edges[k][2]:
                edges[k + 1] = edges[k]
                k -= 1
            edges[k + 1] = key
        return edges

    def mst_kruskals(self):
        disjoint_set = DisjointSetVert()
        for vert in self.vertexes:
            disjoint_set.make_set_vert(vert)
        edges = self.insertion_sort_edges()
        vertex_within_set = 0
        for edge in edges:
            # my union function checks if the sets are within the same set
            add_up = disjoint_set.union(edge[0].get_vertex_index(), edge[1].get_vertex_index())
            if add_up:
                vertex_within_set += 1
                if vertex_within_set == len(self.vertexes):
                    return disjoint_set
        return disjoint_set

    def mst_kruskals_non_vert(self):
        disjoint_set = DisjointSet()
        for vert in self.vertexes:
            disjoint_set.make_set()
        edges = self.insertion_sort_edges()
        for edge in edges:
            # my union function checks if the sets are within the same set
            disjoint_set.union(edge[0].get_vertex_index(), edge[1].get_vertex_index())
        return disjoint_set

    # assumes all edges are one weight
    def mst_kruskals_edge_1(self):
        disjoint_set = DisjointSet()
        for vert in self.vertexes:
            disjoint_set.make_set()
        edges = self.insertion_sort_edges()
        vertex_within_set = 0
        for edge in edges:
            # my union function checks if the sets are within the same set
            add_up = disjoint_set.union(edge[0].get_vertex_index(), edge[1].get_vertex_index())
            if add_up:
                vertex_within_set += 1
                if vertex_within_set == len(self.vertexes):
                    return disjoint_set
        return disjoint_set

    # use distance as key value for prio queue
    def mst_prims_prio(self, vertex_signifier):
        self.init_single_source(vertex_signifier)
        priority_queue = VertexHeap(self.vertexes)
        while priority_queue.is_not_empty():
            vert = priority_queue.extract_min()
            for vert_edge in self.adj_list.get_adjacent_vertex_edges(vert.get_vertex_index()):
                if priority_queue.in_heap(vert_edge[0]) and vert_edge[1] < vert_edge[0].get_distance():
                    vert_edge[0].set_prior(vert)
                    vert_edge[0].set_distance(vert_edge[1])

    def print_vertex_full(self):
        for vert in self.vertexes:
            print(f'Vertex at index {vert.get_vertex_index()}, with signifier {vert.get_vertex_signifier()}.')
            print(f'Prior vertex : {vert.get_prior()}, with distance {vert.get_distance()}.\n')

    def bellman_ford(self, vertex_signifier, print=False):
        self.init_single_source(vertex_signifier)
        edges = self.gather_edges_in_list()
        for i in range(len(self.vertexes) - 1):
            for edge in edges:
                self.relax(edge[0], edge[1], edge[2])
                if print:
                    self.print_vertex_full()
        for edge in edges:
            if edge[1].get_distance() > edge[0].get_distance() + edge[2]:
                return False
        return True



if __name__ == '__main__':
    graph = Graph('graph228.txt', 0)
    #disjoint_set_vert = graph.mst_kruskals()
    #disjoint_set_vert.print()
    #graph.simple_vert_print()
    graph.mst_prims_prio('m')
    #graph.breadth_first_search('u')
    #graph.depth_first_search()
    #graph.print_vert_bfs()
    #graph.print_vert_dfs()
    #graph.print_adj_list()
