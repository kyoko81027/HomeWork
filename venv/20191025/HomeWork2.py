# max 10x + 25y + 50z
# 10x + 25y + 50z <= 3000
# y <= 35
# z <= 20
# x , y , z >= 1
import numpy as np
from scipy.optimize import linprog

# 目標 min(+) max(-)
c = [-10, -25, -50]

# 條件 <=(+) >=(-)
A = [[10, 25 , 50]]
B = np.array([3000])

# 變數範圍
x = (1, None)
y = (1, 20)
z = (1,35)
res = linprog(c, A, B, bounds=(x, y,z))# bounds = 邊界
print(res)