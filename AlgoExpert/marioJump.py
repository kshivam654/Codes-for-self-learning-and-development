m, n = map(int, input().split())
grid = []
for _ in range(m):
    l = [x for x in input()]
    grid.append(l)
totalCal, totalCoin = 0, 0
for j in range(n):
    if grid[m-1][j] == '0':
        maxReach = 0
        count = 0
        for lev in range(m-1, -1, -1):
            if grid[lev][j] == 'C':
                count += 1
                maxReach = max(maxReach, m-1-lev)
        totalCoin += count
        totalCal += 2*maxReach
    else:
        minReach = m-1
        for lev in range(m-1, -1, -1):
            if grid[lev][j] == '0':
                minReach = min(minReach, m-1-lev)
        totalCal += 2*minReach
print('{} {}'.format(totalCoin, totalCal))