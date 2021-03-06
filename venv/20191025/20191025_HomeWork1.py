# max 7x + 5y
# 3x + 4y <= 2400
# 2x + 1y <= 1000
# x >= 100
# y <= 450
# x , y >= 0
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
from scipy.optimize import linprog

# 目標 min(+) max(-)
c = [-7, -5]

# 條件 <=(+) >=(-)
A = [[3, 4], [2, 1]]
B = np.array([2400, 1000])

# 變數範圍
x = (100, None)
y = (0, 450)

res = linprog(c, A, B, bounds=(x, y))# bounds = 邊界
print(res)
