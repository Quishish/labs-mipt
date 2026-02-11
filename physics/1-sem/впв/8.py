import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.special import ellipk
def normalized_period(phi_rad):
    # Аргумент для эллиптического интеграла
    k_squared = np.sin(phi_rad / 2) ** 2
    
    # Вычисление полного эллиптического интеграла 1-го рода
    # Внимание: scipy.ellipk принимает параметр m = k^2
    K = ellipk(k_squared)
    
    # Нормированный период: T/T0 = (2/π) * K(k^2)
    T_T0 = (2 / np.pi) * K
    return T_T0


def scat(name, ln, alph, T0):
    file = open(name+'.txt', 'r')##фух... открыли
    l=ln-1
    T=[0]*(l)
    dT=[0]*(l+1)
    dT[0]=int(list(file.readline()[:-1:].split(","))[3])
    for i in range(l):
        inp=list(file.readline()[:-1:].split(","))
        print(inp)
        dT[i+1]=int(inp[3])
        T[i]=int(inp[4])
    dT=np.array(dT)
    dT=dT/1193180
    sin=[0]*l
    A=[0]*l
    for i in range(1,l+1):
        sin[i-1]=(alph*(1/dT[i-1]+1/dT[i]))**2/16
        A[i-1]=2*(np.arcsin(sin[i-1]**0.5))/math.pi*180
    l1=l-1
    i=0
    while i<l1:
        i+=1
        if T[i]/T0/1193180>10:
            T.pop(i)
            A.pop(i)
            l1-=1
    T=np.array(T)
    A=np.array(A)
    T=T/T0/1193180
    print((normalized_period(A)-T)*T0)
    plt.scatter(A, T, s=4)



degrees = np.linspace(0.1, 175, 500)  # начинаем с 0.1° чтобы избежать деления на 0
radians = np.deg2rad(degrees)

# Вычисление значений
T_T0_theoretical = normalized_period(radians)

# Построение теоретической кривой
plt.plot(degrees, T_T0_theoretical, 
         'b-', linewidth=0.5, label='Теоретическая кривая (без трения)')
scat('1', 391, 0.005208701185, 1.329024761747)
scat('2', 353, 0.005187533711, 1.323623780385)
scat('3', 363, 0.005244989868, 1.338283990919)
scat('4', 332, 0.005320675346, 1.357595498814)
scat('5', 332, 0.005480798796, 1.398451755063)
scat('6', 321, 0.005794607459, 1.478521520664)
scat('7', 311, 0.006285987852, 1.603899553766)



# Настройка графика
plt.xlabel('Амплитуда колебаний φ, градусы', fontsize=14)
plt.ylabel('Нормированный период T/T₀', fontsize=14)
plt.title('Теоретическая зависимость периода от амплитуды\nдля нелинейного маятника (без трения)', 
          fontsize=16, pad=20)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.xlim(0, 180)
plt.ylim(0.9, 5)  # При φ→180 период стремится к бесконечности

# Добавление второй горизонтальной оси с радианами
ax_rad = plt.gca().twiny()
ax_rad.set_xlim(0, np.pi)
ax_rad.set_xlabel('Амплитуда колебаний φ, радианы', fontsize=12)
ax_rad.set_xticks([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])
ax_rad.set_xticklabels(['0', 'π/6', 'π/3', 'π/2', '2π/3', '5π/6', 'π'])



plt.tight_layout()
plt.show()

