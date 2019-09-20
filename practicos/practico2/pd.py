# Se importan las biblotecas que se van a utilizar
import numpy as np
from matplotlib import pyplot as plt
import time
import h5py # es el formato en el que están almacenados los dígitos

from visualizacion import visualizar_conjunto_entrenamiento, visualizar_modelo_lineal
from algoritmos import entrenar_perceptron

plt.rcParams['figure.figsize'] = (10.0, 10.0) # tamaño de las figuras por defecto

X_names = ['TV','radio','prensa escrita']
y_name = ['ventas']

#############################################################################
################  EMPIEZA ESPACIO PARA COMPLETAR  ###########################
#############################################################################

# utilizar la función de numpy loadtxt() para cargar los datos. La primera
# fila no tiene que ser leída



data = np.loadtxt('Advertising.csv', usecols=(0,1,2,3,4), delimiter=',',skiprows=1)
#############################################################################
################  TERMINA ESPACIO PARA COMPLETAR  ###########################
#############################################################################

X=data[:,0:4]
X[:,0]=1
y=data[:,4:]

plt.figure(figsize=(15,5))
for p in range(3):
    plt.subplot(1,3, p+1)
    plt.scatter(X[:, 1+p], y, marker='x', label=X_names[p])
    plt.xlabel('inversión en publicidad de ' + X_names[p])
    plt.ylabel(y_name[0])

def generar_semianillos(N, radio, ancho, separacion):
    '''
    Entrada:
        N: número de muestras a generar
        radio: radio interior del semicírculo
        ancho: diferencia entre el radio exterior e interior
        separación: separación entre los semicírculos
    Salida:
        X: matríz de Nx3 que contiene los datos generados en coordenadas homogéneas
        y: etiquetas asociadas a los datos
    '''

    X = np.ones((N, 3))
    # se sortea a que clase pertenecen las muestras
    y = 2 * (np.random.rand(N) < 0.5) - 1

    # radios y ángulos del semicírculo superior
    radios = radio + ancho * np.random.rand(N)
    thetas = np.pi * np.random.rand(N)
    # coordenadas en x de ambos semicírculos
    X[:,1] = radios * np.cos(thetas) * y + (radio + ancho/2)*(y==-1)
    # coordenadas en y de ambos semicírculos
    X[:,2] = radios * np.sin(thetas) * y - separacion * (y==-1)



    return X, y

N=2000
r = 10; w=5; sep=-5
X, y = generar_semianillos(N, r, w, sep)

visualizar_conjunto_entrenamiento(X, y)

def entrenar_pocket(X, y, w0=None, maxIteraciones = 500):
    """
    Entrada:
        X: matriz de (Nxd+1) que contiene las muestras de entrenamiento
        y: etiquetas asociadas a las muestras de entrenamiento
        w0: inicialización de los pesos del perceptrón
        maxIteraciones: máxima cantidad de iteraciones que el algoritmo puede estar
                        iterando

    Salida:
        w_pocket: parámetros del modelo perceptrón
        error: vector que contiene el error cometido en cada iteración
    """

    if w0 is None:
        # Se inicializan los pesos del perceptrón
        w = np.random.rand(X.shape[1]) # w = np.zeros(d+1)
        print('w inicializado aleatoriamente a ' , w)
    else:
        w = w0
        print('El w incicial es ' , w)

    N = X.shape[0]
    w_pocket = w.copy()  # se inicializa el vector de pesos a devolver
    error = []  # se inicializa la lista de errores
    hayMuestrasMalClasificadas = True
    errorActual = 1  # inicialización del error al máximo posible
    nIter = 0   # inicialización del contador de iteraciones

    while ((nIter < maxIteraciones) and hayMuestrasMalClasificadas):

        #######################################################
        ######## EMPIEZA ESPACIO PARA COMPLETAR CODIGO ########
        #######################################################

        # se calcula el score utilizando los pesos actuales
        score = np.dot(X, w)

        # se encuentran las muestras mal clasificadas
        indicesMalClasificados = y != np.sign(score)

        # se calcula el error en la iteración actual y se lo almacena en
        # la lista de errores
        cantidadMalClasificadas = np.sum(indicesMalClasificados)
        error_i = cantidadMalClasificadas / N

        error.append(error_i)

        if nIter>0 and error[nIter]<error[nIter-1]:
            w_pocket = w.copy()

        if error_i == 0:
            hayMuestrasMalClasificadas = False
        else:
            # si el error es mayor que cero se elige aleatoriamente una de las muestras mal clasificadas
            indice = np.random.randint(cantidadMalClasificadas)
            # y se la utiliza para actualizar los pesos
            w = w + y[indicesMalClasificados][indice] * X[indicesMalClasificados][indice]  # se actualizan los pesos

        nIter = nIter + 1





        #######################################################
        ######## TERMINA ESPACIO PARA COMPLETAR CODIGO ########
        #######################################################

    if(np.array_equal(w, w_pocket)):
        print('Cuidado: es poco probable que el vector de pesos devuelto sea el de la última actualización')


    return w_pocket, error

def transformar_usando_polinomio_de_tercer_grado( X ):
    '''
    Entrada:
        X: matriz de tamaño N x 3 que contiene las características originales en
           coordenadas homogéneas

    Salida:
        Xt: matriz de tamaño N x 10 que contiene las características en el espacio
            transformado

            T(1,x1,x2)= (1, x1, x2, x1^2, x1x2, x2^2, x1^3, x1^2x2, x2^2x1, x2^3)
    '''

    #######################################################
    ######## EMPIEZA ESPACIO PARA COMPLETAR CODIGO ########
    #######################################################
    Xt = np.zeros([X.shape[0],10])

    Xt[:,:3] = X
    Xt[:,3] = X[:,1]**2
    Xt[:,4] = X[:,1]*X[:,2]
    Xt[:,5] = X[:,2]**2
    Xt[:,6] = X[:,1]**3
    Xt[:,7] = X[:,1]**2*X[:,2]
    Xt[:,8] = X[:,2]**2*X[:,1]
    Xt[:,9] = X[:,2]**3

    #######################################################
    ######## TERMINA ESPACIO PARA COMPLETAR CODIGO ########
    #######################################################

    return Xt


# Se transforman las características utilizando el método implementado
Xt = transformar_usando_polinomio_de_tercer_grado( X )
w_inicial = np.zeros(Xt.shape[1])

inicio = time.time()
#######################################################
######## EMPIEZA ESPACIO PARA COMPLETAR CODIGO ########
#######################################################
num_iteraciones = 1000000

w_pocket, error_entrenamiento = entrenar_pocket(Xt, y, w_inicial,num_iteraciones)

#######################################################
######## TERMINA ESPACIO PARA COMPLETAR CODIGO ########
#######################################################
fin = time.time()


print('El algoritmo pocket demoró %f segundos' % (fin - inicio))
print('El pocket finalizó en la iteración %d' % len(error_entrenamiento))
print('El error de entrenamiento es %f' % error_entrenamiento[-1])

plt.figure(figsize=(10,10))
plt.plot(error_entrenamiento[1:],'*-')
plt.xlabel('iteración')
plt.ylabel('error entrenamiento')


def visualizar_frontera_decision(X, y, w):
    '''
    Entrada:
        X: matriz de Nx3 que contiene los puntos en el espacio original
        y: etiquetas de los puntos
        w: vector de tamaño 10 que contiene los parámetros encontrados
    '''

    # Se construye una grilla de 50x50 en el dominio de los datos
    xs = np.linspace( X[:,1].min(), X[:,1].max())
    ys = np.linspace( X[:,2].min(), X[:,2].max())

    XX, YY = np.meshgrid( xs, ys )
    Z = np.zeros_like(XX)

    # se transforman los puntos de la grilla
    pts_grilla = np.vstack( (np.ones(XX.size), XX.ravel(),YY.ravel()) ).T
    pts_grilla_transformados = transformar_usando_polinomio_de_tercer_grado( pts_grilla )

    # los puntos transformados son proyectados utilizando el w
    Z = pts_grilla_transformados @ w
    Z = Z.reshape(XX.shape)#

    # se grafica la frontera de decisión, es decir, la línea de nivel 0
    plt.figure(figsize=(8,8))
    plt.contour(XX, YY, Z, [0])
    plt.scatter(X[:,1][y==1],X[:,2][y==1], s=40, color='b', marker='o',
                label='etiqueta -1')
    plt.scatter(X[:,1][y==-1],X[:,2][y==-1], s=40, color='r', marker='x',
                label='etiqueta 1')
    plt.title('Frontera de decision obtenida mediante transformación no lineal de datos')

visualizar_frontera_decision(X, y, w_pocket)
