import math

x_values = [729986300/10**9]
x_values_log = []

for i in x_values:
    x_values_log.append(round(math.log(i), 3))

print(x_values_log)