def coinGroup(grid):
    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j] == 'C':
            grid[i][j] = 'N'
            return 1 + dfs(i+1,j) + dfs(i-1, j) + dfs(i, j+1) + dfs(i, j-1)
        else: return 0
    m, n = len(grid), len(grid[0])
    ans = [dfs(i, j) for i in range(m) for j in range(n) if grid[i][j]]
    return ans if ans else 0

n = int(input())
grid = []
for _ in range(n):
    l = [x for x in input()]
    grid.append(l)
ans = coinGroup(grid)

ans = [ele for ele in ans if ele != 0]

ans.sort(reverse=True)
player1, player2 = 0, 0
for i in range(len(ans)):
    if i%2 == 0:
        player1 += ans[i]
    else:
        player2 += ans[i]
print('{} {}'.format(player1, player2))