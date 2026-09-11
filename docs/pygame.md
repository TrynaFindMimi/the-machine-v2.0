# Uso de Pygame

## Que es

Pygame se usa en este proyecto unicamente como ventana de renderizado. No maneja logica del juego, solo muestra los frames procesados por OpenCV/MediaPipe.

## Inicializacion

```python
pygame.init()
pygame.display.set_caption("the-machine")
screen = pygame.display.set_mode((w, h))   # tamano = primer frame de la webcam
clock = pygame.time.Clock()
```

## Bucle principal

```python
while running:
    # ... procesar frame ...

    # Convertir frame BGR de OpenCV a surface de Pygame
    rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.transpose(rgb_out, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # Control de framerate
    clock.tick(30)   # 30 FPS
```

## Conversion OpenCV → Pygame

El flujo es: BGR (OpenCV) → RGB → transpose → Surface:

```python
rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
# Pygame espera (width, height, 3) pero OpenCV devuelve (height, width, 3)
surface = pygame.surfarray.make_surface(np.transpose(rgb_out, (1, 0, 2)))
```

## Eventos

```python
for event in pygame.event.get():
    if event.type == pygame.QUIT:           # cerrar ventana
        ...
    if event.type == pygame.KEYDOWN:        # tecla presionada
        if event.key == pygame.K_q:         # q → salir
            ...
        if event.key == pygame.K_n:         # n → siguiente modo
            ...
```

## Teclas disponibles

| Tecla | Accion                         |
|-------|--------------------------------|
| q     | Salir de la aplicacion         |
| n     | Cambiar al siguiente modo      |
