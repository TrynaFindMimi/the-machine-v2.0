# Módulos por capa

> `utils/` migrado a capas. Ver `docs/layers.md` y `docs/architecture.md`.

| Antes (`utils/`) | Ahora | Capa |
|------------------|-------|------|
| `utils/colors.py` | `config/palette.py` + `core/gestures.py` + `presentation/ui/theme.py` | Config / Core / UI |
| `utils/text.py` | `config/strings.py` | Config |
| `utils/fps.py` | `common/fps.py` | Transversal |
| `utils/hand.py` | `presentation/ui/drawing.py` + `presentation/ui/theme.py` | UI |
| `utils/line.py` | `presentation/ui/drawing.py` | UI |
| `utils/perceptron.py` | `core/perceptron.py` | Dominio |
| `utils/results.py` | `core/results.py` | Dominio |
| `utils/style.py` | `presentation/ui/layout.py` + `presentation/ui/effects.py` | UI |

`utils/grid.py` eliminado (modo grid removido).
