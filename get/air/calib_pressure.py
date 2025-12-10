import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

file_calib = open("new_calib0_data.txt")
file_plus = open("new_calib1_data.txt")

def average(file_calib):
    zero_level = 0
    q =0
    for line in file_calib:
        zero_level += int(line[0:6])
        q+=1
    zero_level /= q
    return zero_level

zero_level = average(file_calib)
plus_level = average(file_plus)

colors = ['r','g','b','y','black','pink','v','brown']

dot_atm_pressure = [0,91]
dot_atm_sensor = [zero_level/1000, plus_level/1000]

plt.title("Калибровка датчика давления",fontsize = 18)

plt.scatter(dot_atm_pressure,dot_atm_sensor,color = 'purple',s=20,marker = 'D',label = "Измерения")
plt.plot(dot_atm_pressure,dot_atm_sensor,label ="график прямой y = 23.6x + 214876")

ax = plt.gca()
ax.set_xlabel("Давление (Па)", fontsize=18)    # +
ax.set_ylabel("АЦП (у.е.) $\cdot10^{3}$", fontsize=18)

plt.legend(fontsize = 18)
plt.minorticks_on()
plt.grid(which='major', color = '#444', linewidth = 1)
plt.grid(which='minor', color='#aaa', ls=':')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
#print(arr00[0:10])