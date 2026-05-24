# Agente MCTS para Connect-4 — Juan (Grupo A)

## ¿Qué hace este agente?

Este agente usa el algoritmo **Monte Carlo Tree Search (MCTS)** para jugar Connect-4. La idea es simple: en lugar de calcular todas las jugadas posibles (lo cual sería demasiado lento), el agente simula muchas partidas aleatorias desde el estado actual y usa esos resultados para estimar qué columna es la mejor opción.

Con cada simulación que hace, el agente aprende un poco más sobre qué movimientos llevan a ganar. Al final del tiempo disponible, elige la columna que mejor resultado tuvo en todas esas pruebas.

---

## Versiones del agente

### V1 — MCTS Base (`policyV1.py`)
Enlace al código: [`policyV1.py`](./tournament/groups/Group%20A/policyV1.py)

La primera versión implementa MCTS de forma directa:
- Guarda estadísticas de cada jugada en un diccionario (`stats`), con recompensa acumulada y número de visitas por columna.
- Usa la fórmula **UCT** para balancear exploración (probar columnas nuevas) con explotación (preferir las que ya funcionaron).
- Al llegar a un estado desconocido, simula una partida aleatoria hasta el final y propaga el resultado de vuelta.
- **Optimizaciones clave:**
  - Reemplazó `np.sqrt` y `np.log` con `math.sqrt` y `math.log` para operaciones sobre números individuales — esto redujo el tiempo por nodo en ~40%.
  - Límite combinado de tiempo (50 ms) + máximo de 150 iteraciones para garantizar respuestas rápidas sin timeouts.
  - La recompensa se asigna por jugador en cada nodo (no al agente global), lo que evita sesgos en un juego de suma cero. Los empates valen 0.5.

### V2 — MCTS con Filtros Heurísticos (`policyV2.py`)
Enlace al código: [`policyV2.py`](./tournament/groups/Group%20A/policyV2.py)

La segunda versión agrega dos revisiones rápidas **antes** de arrancar el árbol MCTS:

1. **Filtro de ataque:** Si hay una columna que le da la victoria inmediata al agente, la elige directamente sin simular nada.
2. **Filtro de defensa:** Si no hay victoria inmediata propia, revisa si el rival puede ganar en su próximo turno y lo bloquea.

Además, el rollout de la V2 no es completamente aleatorio: en cada paso de la simulación también revisa si hay una jugada ganadora disponible. Esto hace que las simulaciones sean más informativas.

**¿Por qué esto mejora el desempeño?** El MCTS puro a veces se "pierde" jugadas ganadoras obvias si sus simulaciones aleatorias no las encuentran con frecuencia suficiente. Resolver esos casos con lógica directa libera el árbol para concentrarse en la estrategia a mediano plazo.

---

## Análisis de rendimiento

Se corrieron 100 partidas por configuración. Los resultados principales son:

| Agente | vs. Aleatorio (rojo) | vs. Aleatorio (amarillo) | vs. sí mismo |
|--------|----------------------|--------------------------|--------------|
| V1     | ~95% victorias       | ~93% victorias           | ~50% cada uno |
| V2     | ~97% victorias       | ~96% victorias           | ~50% cada uno |

- **V1 vs Q-Learning (Grupo B):** ~65% de victorias. Mostraba vulnerabilidades en situaciones de amenaza directa.
- **V2 vs Q-Learning (Grupo B):** ~88% de victorias. Los filtros de ataque/defensa eliminaron esos errores.

El comportamiento del agente varía según el parámetro `c_exploration` (constante de exploración en UCT):
- Valores bajos (`c ≈ 0.7`) hacen al agente más "explotador" — funciona bien cuando el tiempo es limitado.
- Valores altos (`c ≈ 1.41`) lo hacen más exploratorio — mejor cuando hay más tiempo de cómputo disponible.

---

## Propuesta de mejora (versión futura)

La debilidad más clara es que las simulaciones (rollouts) siguen siendo bastante aleatorias. Incluso en la V2, la gran mayoría de las partidas simuladas terminan de forma poco realista.

**Mejora propuesta:** Agregar una **función de evaluación heurística del tablero** para no tener que llegar siempre al final de cada simulación. Por ejemplo, contar cuántas "amenazas" activas tiene cada jugador (secuencias de 3 fichas con espacio libre). Esto permitiría hacer simulaciones más cortas pero más precisas, aumentando el número de iteraciones útiles por jugada sin cambiar el límite de tiempo.

Esta mejora tendría un impacto alto porque el cuello de botella actual no es la velocidad de selección/propagación sino la calidad de la información que aporta cada simulación.

---

## Instalación y uso

### Requisitos
```
python >= 3.10
numpy
```

El entorno de Connect-4 debe estar disponible como paquete `connect4` en el path de Python (provisto por el repositorio del grupo).

### Ejecutar el agente

```python
# Desde la raíz del branch
import sys
sys.path.append("tournament/groups/Group A")
from policyV2 import MCTSAgentV2

agente = MCTSAgentV2(time_limit=0.15, c_exploration=1.414)
accion = agente.act(estado_del_tablero)
```

Para el autocalificador (Gradescope), el método `mount(timeout)` ajusta automáticamente el tiempo disponible:

```python
agente.mount(timeout=0.2)  # Ajusta el límite al 85% del timeout del servidor
```

### Reproducibilidad

Para obtener resultados reproducibles en pruebas locales, se puede fijar la semilla del generador de números aleatorios:

```python
agente = MCTSAgentV2(rng_seed=42)
```

---

## Estructura del repositorio

```
Juan-Montes/                          # Branch personal
├── tournament/
│   ├── groups/
│   │   ├── Group A/
│   │   │   ├── policyV1.py           # Agente V1: MCTS base
│   │   │   └── policyV2.py           # Agente V2: MCTS + filtros heurísticos
│   │   ├── Group B/                  # Agentes del compañero de grupo
│   │   └── Group C/                  # Agentes del otro compañero
│   └── versus/
│       ├── match_Group_A_vs_Group_B.json   # Resultados torneo A vs B
│       └── match_Group_C_vs_Group_A.json   # Resultados torneo C vs A
├── entrega.ipynb                     # Notebook con experimentos y gráficas
├── informe.tex                       # Fuente LaTeX del informe
├── informe.pdf                       # Informe técnico compilado
├── main.py                           # Script principal para correr partidas
├── tournament.py                     # Lógica del torneo entre agentes
├── resultados.png                    # Gráfica de tasas de victoria
└── README.md                         # Este archivo
```
