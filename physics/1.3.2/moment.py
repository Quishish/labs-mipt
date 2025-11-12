x = [50, 100, 150, 200, 250, 300, 400, 500]

g = 9.8

d = 5.5/100

ans = []

for i in range (len(x)):
    m = x[i]/1000
    mom = m * g * d
    ans.append(mom)

print(ans)