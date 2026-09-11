# Creditos

## Musica

El modo music reproduce las canciones de **EPIC: The Musical**, obra compuesta por **Jorge Hernan**.

La musica se incluye aqui con fines estrictamente academicos y educativos dentro del proyecto de Inteligencia Artificial "the-machine". **Este proyecto no genera ningun tipo de beneficio economico** y no debe usarse con animo de lucro.

- Titulo: EPIC: The Musical
- Autor/compositor: Jorge Hernan
- Uso: solo academico (proyecto universitario de deteccion de gestos)

Si usas, distribuyes o modificas este repositorio, conserva esta atribucion y la nota de uso no comercial.

## Modelos

Los pesos `models/finger_1..4.npz` son producto del entrenamiento propio (`controllers/hand.FingerRNN`) sobre caracteristicas sinteticas generadas en el proyecto; no derivan de material con copyright.

Los pesos `models/thumb.npz` son producto del entrenamiento propio (`controllers/thumb.ThumbPerceptron`, `core/perceptron`) sobre polos sinteticos del gesto del pulgar; no derivan de material con copyright.

Los modelos `.task` de MediaPipe (`models/hand_landmarker.task`, `models/gesture_recognizer.task`) pertenecen a Google LLC y se distribuyen bajo los terminos del proyecto MediaPipe.