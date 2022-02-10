# O(v^2+e) time | O(v) space 
def dijkstrasAlgorithm(start, edges):
    numberOfVertices = len(edges)

    minDistances = [float('inf') for _ in range(numberOfVertices)]
    minDistances[start] = 0

    visited = set()

    while len(visited) != numberOfVertices:
        vertex, currentMinDistance = getVertexWithMinDistance(minDistances, visited)

        if currentMinDistance == float('inf'):
            break

        visited.add(vertex)
        for edge in edges[vertex]:
            destination, distanceToDestination = edge

            if destination in visited:
                continue
            newPathDistance = currentMinDistance + distanceToDestination
            currentDestinationDistance = minDistances[destination]

            if newPathDistance < currentDestinationDistance:
                minDistances[destination] = newPathDistance
    return minDistances

def getVertexWithMinDistance(distances, visited):
    currentMin = float('inf')
    vertex = None

    for vertexIdx, distance in enumerate(distances):
        if vertexIdx in visited:
            continue
        if distance <= currentMin:
            currentMin = distance
            vertex = vertexIdx
    return vertex, currentMin