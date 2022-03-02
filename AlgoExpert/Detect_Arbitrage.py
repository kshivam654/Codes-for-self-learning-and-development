# DETECT ARBITRAGE

import math

# O(N^3) time and O(N^2) space
def detectArbitrage(exchangeRates):
    # Write your code here.
	logExchangeRates = convertToLogMatrix(exchangeRates)
	return foundNegativeWeightCycle(logExchangeRates, 0)

def foundNegativeWeightCycle(graph, start):
	distancesFromStart = [float("inf") for _ in range(len(graph))]
	distancesFromStart[start] = 0
	
	for _ in range(len(graph) - 1):
		if not relaxEdgesAndUpdateDistances(graph, distancesFromStart):
			return False
		
	return relaxEdgesAndUpdateDistances(graph, distancesFromStart)

def relaxEdgesAndUpdateDistances(graph, distances):
	updated = False
	for sourceIdx, edges in enumerate(graph):
		for destinationIdx, edgeWeight in enumerate(edges):
			newDistanceToDestination = distances[sourceIdx] + edgeWeight
			if newDistanceToDestination < distances[destinationIdx]:
				updated = True
				distances[destinationIdx] = newDistanceToDestination
				
	return updated

def convertToLogMatrix(matrix):
	newMatrix = []
	for row, rates in enumerate(matrix):
		newMatrix.append([])
		for rate in rates:
			newMatrix[row].append(-math.log10(rate))
			
	return newMatrix


#My approach: Bellman's fort algorithm
#Time: O(E*K) and Space: O(V)
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        prices = [float('inf') for _ in range(n)]
        prices[src] = 0
        for i in range(K+1):
            tempPrice = prices.copy()
            for s,d,p in flights:
                if prices[s] == float('inf'):
                    continue
                if prices[s]+p < tempPrice[d]:
                    tempPrice[d] = prices[s]+p
            prices = tempPrice
        return prices[dst] if prices[dst] != float('inf') else -1