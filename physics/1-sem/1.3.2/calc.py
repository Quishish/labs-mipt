import math

x1 = [0, 18.67, 41.67, 64.5, 82.67, 105.67, 152.33, 195.67]
x2 = [195.67, 171.00, 119.83, 94.00, 69.33, 45.33, 24.5, 0.37]

l = 1565

phi1 = []
phi2 = []

for i in range (len(x1)):
    x = round(math.atan(x1[i]/l), 6)
    y = round(math.atan(x2[i]/l), 6)
    phi1.append(x)
    phi2.append(y)

print(phi1)
print(phi2)