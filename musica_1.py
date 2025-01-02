import pygame
import numpy as np

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)

# Función para generar un tono
def generar_tono(frecuencia, duracion, volumen=0.5):
    # Cálculo de frames
    frames = int(duracion * 44100)
    arr = np.array([4096 * volumen * np.sin(2.0 * np.pi * frecuencia * x / 44100) for x in range(frames)]).astype(np.int16)
    arr2 = np.c_[arr,arr]
    return pygame.sndarray.make_sound(arr2)

# Definir las frecuencias de las notas
notas = {
    'DO': 261.63,
    'RE': 293.66,
    'MI': 329.63,
    'FA': 349.23,
    'SOL': 392.00
}

# Duración de cada nota (en segundos)
duracion = 0.5

# Crear la melodía
melodia = ['DO', 'RE', 'MI', 'FA', 'SOL', 'FA', 'MI', 'RE']

# Generar y reproducir la melodía
for nota in melodia:
    tono = generar_tono(notas[nota], duracion)
    tono.play()
    pygame.time.wait(int(duracion * 1000))  # Esperar en milisegundos

pygame.quit()