# O(v + e) time | O(v) space
def cycleInGraph(edges):
    numberOfNodes = len(edges)
    visited = [False for _ in range(numberOfNodes)]
    currentInStack = [False for _ in range(numberOfNodes)]

    for node in range(numberOfNodes):
        if visited[node]:
            continue
        containCycle = isNodeInCycle(edges, node, visited, currentInStack)
        if containCycle:
            return True
    return False

def isNodeInCycle(edges, node, visited, currentInStack):
    visited[node] = True

    currentInStack[node] = True
    neighbors = edges[node]
    for neighbor in neighbors:
        if not visited[neighbor]:
            containsCycle = isNodeInCycle(edges, node, visited, currentInStack)
            if containsCycle:
                return True
        elif currentInStack[neighbor]:
            return True
    
    currentInStack[node] = False
    return False