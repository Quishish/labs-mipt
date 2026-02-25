import matplotlib.pyplot as plt
import numpy as np
#погрешности (здесь они настолько малы, что не видны на графике)
df = [1]*5
dk = [0]*5


#функция для метода наименьших квадратов линейной функции
def mnk(x, y):
    k=(sum(x*y)-sum(x)*sum(y)/np.size(x))/(sum(x**2)-sum(x)**2/np.size(x))
    b=(sum(y)-k*sum(x))/np.size(y)
    print(k)
    return [k, b]

def dmnk(x,y,a,b):
    x1=sum(x)/np.size(x)
    dxx=0
    for i in range(np.size(x)):
        dxx+=(x[i]-x1)**2
    dxx=dxx/np.size(x)
    y1=sum(y)/np.size(y)
    dyy=0
    for i in range(np.size(y)):
        dyy+=(y[i]-y1)**2
    dyy=dyy/np.size(y)
    delk=(1/(np.size(x)-2)*(dyy/dxx-a**2))**0.5
    delb=delk*(sum(x**2)/np.size(x))**0.5
    print(delk)
    return(delk)   
#f1 = [244, 494, 744, 992, 1238]
#f1 = [248, 503, 756, 1007, 1257]
#f1 = [251, 512, 768, 1023, 1277]
f1 = [265, 519, 780, 1038, 1297] #пересчитанные данные с вольтметра (со 150 до 600)
f1 = np.array(f1) #для дальнейшего удобства в массив numpy
k1 = [1,2,3,4,5] #данные с амперметра
k1 = np.array(k1) #так как изначально амперметр показал значение силы тока на параллельном соединении, а вольтметр на проволке, то пересчитаем силлу тока на проволке
plt.errorbar(k1, f1, xerr=dk, yerr=df, fmt="o", color="r", capsize=0.1)#отображение точек с погрешностями (если увеличить график, то они все же будут видны)
plt.title('График зависимости резонансной частоты от номера резонанса', fontsize=20)#добавляем название графику
plt.xlabel("Номер резонанса", fontsize=20)
plt.ylabel("Частота резонанса, Гц", fontsize=20)
s1=mnk(k1,f1)#получаем массив, где первый элемент k,  второй b (y=kx+b)
dmnk(k1,f1, s1[0], s1[1])
x1=[1,5]#заводим две точки для прямой
y1=[1*s1[0]+s1[1],5*s1[0]+s1[1]]#считаем значения прямой в этих точках
plt.plot(x1,y1, label=f'прямая по мнк с угловым коэффициентом {s1[0]:.1f}')#наносим прямую и задаем ей название
plt.minorticks_on() # Включить мелкие деления
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray') # Мелкая сетка
plt.show()#воспроизведение всех графиков на экран
