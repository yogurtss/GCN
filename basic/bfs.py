# 广度优先算法
#  利用队列先进先出的方式
import collections


class Solution:
    def maxAreaOfIsland(self, grid):
        res = 0
        for i, l in enumerate(grid):
            for j, n in enumerate(l):
                queue = collections.deque([(i, j)])
                cur_area = 0
                while queue:
                    cur_i, cur_j = queue.popleft()
                    if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(l) or grid[cur_i][cur_j] != 1:
                        continue
                    cur_area += 1
                    grid[cur_i][cur_j] = 0
                    for p, q in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                        next_i, next_j = cur_i + p, cur_j + q
                        queue.append((next_i, next_j))
                res = max(res, cur_area)
        return res


if __name__ == '__main__':
    input = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]
    s = Solution()
    res = s.maxAreaOfIsland(input)
    print(res)
