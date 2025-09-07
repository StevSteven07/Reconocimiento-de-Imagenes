import cv2
import numpy as np
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os


"""Funcion Seleccionar Imagenes"""
def seleccionar_imagenes():
    """Funsion usada para seleccionar imagenes en interfaz grafica
    abre explorador de archivos"""
    root= tk.Tk()
    root.withdraw() #oculta la ventana de tkinter

    #Dialogo para la primera imagen
    print("Selecciona la primera imagen")
    img1_path= filedialog.askopenfilename(
        title="Seleccionar primera imagen",
        filetypes=[
            ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Todos los archivos","*.*")
        ]
    )

    if not img1_path:
        print("Se cancelo la seleccion de la primera imagen")
        return None, None

    #Dialogo de la segunda imagen
    print("Seleccionar la segunda imagen"),
    img2_path= filedialog.askopenfilename(
          filetypes=[
              ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
              ("Todos los archivos", "*.*")
          ]
    )

    if not img2_path: 
        print("Se cancelo la seleccion de la segunda imagen")
        return None, None
    
    return img1_path, img2_path

img1_path, img2_path= seleccionar_imagenes()

def comparar_imagenes(img1_path, img2_path):
    # Cargar imágenes en escala de grises para facilitar la detección
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Error: No se pudieron cargar las imagenes.")
        return
    
    # Crear el detector ORB
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Detectar puntos clave y descriptores
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Crear el comparador de características Brute Force con Hamming
    bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Encontrar coincidencias aproximadas
    matches= bf.knnMatch(des1, des2, k=2)

    # Aplicar ratio test de Lowe para filtrar buenas coincidencias
    buenas= []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            buenas.append(m)

    # Calcular porcentaje de similitud basado en puntos clave encontrados
    similitud=(len(buenas) / max(len(kp1), len(kp2))) * 10000

    #Clasificacion del porcentaje de similitud
    if similitud >= 90:
        categoria= "Imagenes iguales ✅"
    elif similitud >=50:
        categoria= "Imagenes Similares 🔄"
    else:
        categoria= "Imagenes Diferentes ❌"

    # Mostrar resultados
    print(f"Coincidencias totales: {len(matches)}")
    print(f"Buenas coincidencias: {len(buenas)}")
    print(f"Porcentaje de similitud: {similitud:.0f}%")
    print(f"Resultado: {categoria}")

    """# Dibujar las coincidencias en una imagen
    img_coincidencias= cv2.drawMatches(img1, kp1, img2, kp2, buenas, None, flags=2)

    # Mostrar imagen resultante
    plt.figure(figsize=(15,8))
    plt.imshow(img_coincidencias)
    plt.title(f"Similitud: {similitud:.0f}%")
    plt.axis("off")
    plt.show() """

    return similitud
# Ruta de tus imágenes
#imagen1= r"C:\Users\PC\Desktop\Steven\Conocimientos\Informatica\Proyectos\IMGDupl\img\foto_prueba.jpg"
#imagen2= r"C:\Users\PC\Desktop\Steven\Conocimientos\Informatica\Proyectos\IMGDupl\img\2017-02-08 18.52.04.png"

#seleccionar_imagenes()
comparar_imagenes(img1_path, img2_path)