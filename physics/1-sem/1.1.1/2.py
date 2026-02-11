x_data1 = [30, 35, 40, 55, 70, 85, 101, 115, 131, 146]
x_data2 = [12, 15, 30, 45, 60, 75, 90, 105, 121, 140]
x_data3 = [10, 20, 30, 40, 50, 65, 79, 94, 112, 129]

x1_data1 = []
x2_data2 = []
x3_data3 = []

for i in x_data1:
    x1_data1.append(round(i*5, 2))
    
for i in x_data2:
    x2_data2.append(round(i*5, 2))

for i in x_data3:
    x3_data3.append(round(i*5, 2))

print(x1_data1)
print(x2_data2)
print(x3_data3)