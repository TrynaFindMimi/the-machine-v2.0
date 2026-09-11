# Uso de OpenCV (cv2)

## Que es

OpenCV es la libreria principal para manipulacion de imagenes y video. En este proyecto se usa para captura de webcam, dibujo de primitivas y conversion de color.

## Captura de video

```python
vc = cv2.VideoCapture(0)          # 0 = camara por defecto
ret, frame = vc.read()            # ret=True si leyo bien, frame=np.array BGR
vc.release()                      # liberar recurso al salir
```

## Formato de color

OpenCV usa **BGR** por defecto (no RGB). MediaPipe y Pygame necesitan RGB, asi que se convierte:

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # BGR → RGB para MediaPipe
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)     # RGB → BGR si se necesita volver
```

## Primitivas de dibujo

```python
# Linea
cv2.line(frame, (x1,y1), (x2,y2), color, thickness, cv2.LINE_AA)

# Circulo
cv2.circle(frame, (cx,cy), radius, color, thickness, cv2.LINE_AA)
# thickness=-1 → circulo relleno

# Rectangulo
cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

# Texto
cv2.putText(frame, "texto", (x,y), font, scale, color, thickness, cv2.LINE_AA)

# Tamano del texto (para centrar/posicionar)
(tw, th), baseline = cv2.getTextSize("texto", font, scale, thickness)
```

## Transformaciones

```python
# Voltear horizontalmente (espejo)
frame = cv2.flip(frame, 1)

# Mezclar dos imagenes con transparencia
cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
```

## Constantes de estilo

- `cv2.LINE_AA` → anti-aliasing (lineas suaves)
- `cv2.FONT_HERSHEY_SIMPLEX` → fuente sans-serif usada en todo el proyecto
