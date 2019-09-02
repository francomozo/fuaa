#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 07:52:52 2019

@author: fuaa
"""
import numpy as np

def entrenar_perceptron(X, y, w0=None, maxIteraciones = 500):
    """
    Entrada:
        X: matríz de (Nxd+1) que contiene las muestras de entrenamiento
        y: etiquetas asociadas a las muestras de entrenamiento
        maxEpoca: máxima cantidad de épocas que el algoritmo puede estar iterando
        recorridoEstocastico: indica si el orden en que son recorridas las muestras
                              varía de una época a otra
        w_inicial: inicialización de los pesos del perceptrón
        
    Salida:
        w: parámetros del modelos perceptrón   
        error: vector que contiene el error cometido en cada iteración
    """
    
    if w0 is None:
        # Se inicializan los pesos del perceptrón
        w = np.random.rand(X.shape[1]) # w = np.zeros(d+1)
    else:
        w = w0
 
    
    nIter = 0    
    error = []
    N = X.shape[0]
    hayMuestrasMalClasificadas = True
    while ((nIter < maxIteraciones) and hayMuestrasMalClasificadas):
    
        # se calcula el score utilizando los pesos actuales
        score = np.dot(X, w)   
        
        # se encuentran las muestras mal clasificadas
        indicesMalClasificados = y != np.sign(score) 

        # se calcula el error en la iteración actual y se lo almacena en 
        # la lista de errores
        cantidadMalClasificadas = np.sum(indicesMalClasificados)
        error_i = cantidadMalClasificadas / N
        
        error.append(error_i)
        
        if error_i == 0:
            hayMuestrasMalClasificadas = False     
        else:           
            # si el error es mayor que cero se elige aleatoriamente una de las muestras mal clasificadas
            indice = np.random.randint(cantidadMalClasificadas) 
            # y se la utiliza para actualizar los pesos
            w = w + y[indicesMalClasificados][indice] * X[indicesMalClasificados][indice]  # se actualizan los pesos
        
        nIter = nIter + 1
    
    return w, error

#####################################################################################
