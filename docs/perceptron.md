# Perceptron — Modo Line (`core/perceptron.py`)

Entrena perceptron binario en vivo entre P5 (idx 4 thumb tip) y P9 (idx 8 index tip), 2 manos con `_perc[0]/[1]` persistentes.

`build_dataset(p5,p9)`: 24 puntos sobre segmento + 24 desplazados perp `MARGIN=0.03` → `X=[x,y,1]`, `y=±1`.

`Perceptron.train_budget(X,y,200)`: 200 épocas max/frame, converge en 1-5 tras warmup. `FINGERS_TOGETHER_THRESH=0.04` resetea si dedos juntos.

Import: `from core.perceptron import Perceptron, build_dataset, FINGERS_TOGETHER_THRESH`. Visual: `presentation/ui/drawing.draw_line` azul `1px`.

