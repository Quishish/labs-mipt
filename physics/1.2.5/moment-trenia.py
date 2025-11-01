import math

n = [4, 6, 10, 14]
m = [76, 112, 215, 335]
a = [0.00032, 0.00031, 0.00036, 0.00040]

ress = []

for i in range(4):
    mt = (m[i] * 9.8 * 0.121 * a[i]) / (2 * math.pi * n[i])
    ress.append(mt)

print(ress)
print(sum(ress)/4)