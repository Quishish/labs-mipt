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
         'b-', linewidth=0.5, label='Теоретическая кривая ')
scat('ELP_005', 294, 0.005215511288, 1.330762391729)
scat('ELP_010', 294, 0.005817129952, 1.484268241483)
scat('ELP_015', 294, 0.005236084900, 1.336011846190)
scat('ELP_020', 294, 0.005180148855, 1.321739499393)
scat('ELP_025', 294, 0.005388244342, 1.374836048041)


# Настройка графика
plt.xlabel('Амплитуда колебаний φ, градусы', fontsize=20)
plt.ylabel('Нормированный период T/T₀', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.title('Теоретическая зависимость периода от амплитуды\nдля нелинейного маятника', 
          fontsize=30, pad=20)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=14)
plt.xlim(0, 180)
plt.ylim(0.9, 3.5)  # При φ→180 период стремится к бесконечности

# Добавление второй горизонтальной оси с радианами
ax_rad = plt.gca().twiny()
ax_rad.set_xlim(0, np.pi)
ax_rad.set_xlabel('Амплитуда колебаний φ, радианы', fontsize=20)
ax_rad.set_xticks([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])
ax_rad.set_xticklabels(['0', 'π/6', 'π/3', 'π/2', '2π/3', '5π/6', 'π'])



plt.tight_layout()
plt.show()

