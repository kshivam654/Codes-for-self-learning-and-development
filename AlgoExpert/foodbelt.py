m, n = map(int, input().split())
matrix = []
for i in range(m):
    l = list(map(str, input().split()))
    if i%2 == 0:
        matrix.extend(l)
    else:
        matrix.extend(l[::-1])

def rotate(matrix, r):
    end = len(matrix)
    k = matrix[end-r: end] + matrix[0:end-r]
    return k
r = int(input())
tempAns = rotate(matrix, r)
f = input()
ans = []
res = [tempAns[i:i+n] for i in range(0,len(tempAns),n)]
# print(res)
for i in range(m):
    if i%2 == 0:
        print(' '.join(res[i]))
        ans.append(res[i])
    else:
        print(' '.join(res[i][::-1]))
        ans.append(res[i][::-1])
flag = 0
fi, fj = 0, 0
for i in range(m):
    for j in range(n):
        if ans[i][j] == f:
            fi, fj = i, j
            flag = 1
            break
if flag:
    print('[{}, {}]'.format(fi, fj))
else:
    print('Not Available')