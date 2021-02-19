# 深度优先算法两种实现过程
# 1. 栈
# 2. 递归


class Solution:
    def maxAreaOfIsland(self, grid):
        res = 0
        for i, l in enumerate(grid):
            for j, n in enumerate(l):
                stack = [(i, j)]
                cur_area = 0
                while stack:
                    cur_i, cur_j = stack.pop()
                    if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(l) or grid[cur_i][cur_j] != 1:
                        continue
                    cur_area += 1
                    grid[cur_i][cur_j] = 0
                    for p, q in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                        next_i, next_j = cur_i + p, cur_j + q
                        stack.append((next_i, next_j))
                res = max(res, cur_area)
        return res

    def maxAreaOfIsland_(self, grid):
        res = 0
        for i, l in enumerate(grid):
            for j, n in enumerate(l):
                res = max(self.dfs(i, j, grid), res)
        return res

    def dfs(self, i, j, grid):
        if i < 0 or j < 0 or i == len(grid) or j == len(grid[0]) or grid[i][j] != 1:
            return 0
        ans = 1
        grid[i][j] = 0
        for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            next_i, next_j = i + dx, j + dy
            ans += self.dfs(next_i, next_j, grid)
        return ans


if __name__ == '__main__':
    input = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]
    s = Solution()
    res = s.maxAreaOfIsland_(input)
    print(res)
