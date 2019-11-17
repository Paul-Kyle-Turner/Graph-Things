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
        self.heap[0] = self.heap.pop(len(self.heap) - 1)
        self.perc_down_min(0)
        return min_element

    def insert_element(self, key):
        self.heap.append(key)
        index = len(self.heap) - 1
        while index > 0 and self.parent(index) > self.heap[index]:
            self.heap[index], self.heap[self.parent_index(index)] = self.parent(index), self.heap[index]
            index = self.parent_index(index)


class Link:

    def __init__(self, value, link=None):
        self.value = value
        self.link = link

    def set_link(self, link):
        self.link = link

    def get_value(self):
        return self.value

    def get_link(self):
        return self.link


class LinkedList:

    def __init__(self, head=None):
        self.head = None
        self.length = 0

    def insert_at_head(self, value):
        self.head = Link(value, self.head)
        self.length += 1

    def get_length(self):
        return self.length

    def pop_head(self):
        temp = self.head
        self.head = self.head.get_link()
        self.length -= 1
        return temp

    def get_item_at_pos(self, pos):
        temp = self.head
        if pos < self.length:
            for i in range(pos):
                temp = temp.get_link()
            return temp.get_value()

    def inset_item_at_pos(self, value, pos):
        temp = self.head
        if pos < self.length:
            for i in range(pos - 1):
                temp = temp.get_link()
            next_link = temp.get_link()
            new_link = Link(value, next_link)
            temp.set_link(new_link)


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


class Graph:

    def __init__(self, file_path, signifier_index=None):
        self.adj_matrix = self.adjacent_matrix_from_file(file_path, signifier_index)
        self.adj_list = []
        self.vertexes = []
        self.adjacent_list_from_file(file_path, signifier_index)
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
                self.adj_list.append([])
                if signify_index is not None:
                    self.vertexes[i].set_vertex_signifier(line_list.pop(signify_index))
                for k in range(n):
                    if int(line_list[k]) != 0:
                        self.adj_list[i].append(self.vertexes[k])
                i += 1
        return self.adj_list

    def print_adj_list(self):
        i = 0
        for ind in self.adj_list:
            print(f'STARTING WITH INDEX {i}')
            for inner in ind:
                print(inner.get_vertex_index())
                print(inner.get_vertex_signifier())
            i += 1

    def print_adj_matrix(self):
        print(self.adj_matrix)

    def get_vertex_with_sig(self, signifier):
        for vert in self.vertexes:
            if vert.get_vertex_signifier == signifier:
                return vert
        return None

    def get_adjacent_vert(self, index):
        return self.adj_list[index]

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
        print(self.vertexes)

    def depth_first_search(self):
        for vert in self.vertexes:
            vert.set_color('white')
            vert.set_prior(None)
        self.set_time_zero()
        for vert in self.vertexes:
            if vert.get_color() == 'white':
                self.depth_first_visit(vert)

    def depth_first_visit(self, vertex):
        vertex.set_distance(self.time_plus_one())
        vertex.set_color('gray')
        for vert in self.get_adjacent_vert(vertex.get_vertex_index()):
            if vert.get_color() == 'white':
                vert.set_prior(vertex)
                self.depth_first_visit(vert)
        vertex.set_color('black')
        vertex.set_vertex_end_time(self.time_plus_one())

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


if __name__ == '__main__':
    graph = Graph('graph226.txt', 0)
    #graph.breadth_first_search('u')
    graph.depth_first_search()
    #graph.print_vert_bfs()
    graph.print_vert_dfs()
