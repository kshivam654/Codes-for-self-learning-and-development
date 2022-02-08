import queue


class Node:
    def __init__(self, name) -> None:
        self.name = name
        self.children = []

    def add(self, name):
        self.children.append(Node(name))
    
    # O(v + e) time | O(v) space
    def breathFirstSearch(self, array):
        queue = [self]
        while len(queue):
            current = queue.pop(0)
            array.append(current.name)
            for child in current.children:
                queue.append(child)
        return array