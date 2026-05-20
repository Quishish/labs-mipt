import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

def fun(arrx,arry,ch):
    x = arrx
    y = arry

    xy_av = 0
    x_2_av = 0
    y_2_av = 0
    x_av=0
    y_av = 0
    for i in range(len(arrx)):
        x_av += x[i]
        y_av += y[i]
        xy_av += x[i] * y[i]
        x_2_av += x[i] * x[i]
        y_2_av += y[i] * y[i]
    
    x_av /= len(x)
    y_av /= len(y)
    xy_av /= len(x)
    x_2_av /= len(x)
    y_2_av /= len(x)

    a = (xy_av - x_av * y_av)/(x_2_av - x_av ** 2)
    b = y_av - a * x_av

    sigma_b = 1/((len(x))**0.5) * ( (y_2_av - y_av**2) / (x_2_av - x_av ** 2) - b**2  ) ** 0.5
    sigma_a = sigma_b * (x_2_av - x_av**2)**0.5

    hi = 0
    for i in range(len(arrx)):
        hi += (arry[i] - (a*arrx[i]+b))**2 / (0.03)**2
    if(ch):
        print(("Hi: ", hi**0.5, len(arrx)))
        print("K: ", a)
        print(sigma_a)

    return(a,b,hi**0.5)
    
t = np.array([2*60+7, 2*60+23, 3*60+10, 5*60+7])
V = np.array([26, 26, 25, 25])
h = np.array([56, 49, 40, 28])