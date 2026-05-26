# Agente Connect-4 — Nicolas Almonacid

Agente para el reto final de Fundamentos de Inteligencia Artificial (2026.1).

**Enfoque:** Q-Learning Bipolar con aprendizaje offline en self-play.

## Estructura

```
nico_agent/
├── policy_v0.py          V0: politica aleatoria (baseline para comparar)
├── policy_v1.py          V1: Monte Carlo Control + eps-greedy (Hoja 11)
├── policy_v2.py          V2: Q-Learning Bipolar + UCB + heuristica (Hojas 10-12)
├── qtable.py             Estructura compartida (estado canonicalizado por jugador)
├── heuristic.py          Reglas de seguridad: WIN / BLOCK / preferencia central
├── harness.py            Evaluacion: 1 partida, N partidas, desagregado por color
├── train_v1.py           Script de entrenamiento de V1
├── train_v2.py           Script de entrenamiento de V2
├── entrega.ipynb         Notebook con los 6 experimentos del analisis
├── data/
│   ├── qtable_v1.pkl     Q-table entrenada para V1
│   └── qtable_v2.pkl     Q-table entrenada para V2 (la del torneo)
└── readme.md             Este archivo
```

## Cómo se conecta con el framework del torneo

El agente final es `QLearningPolicy` definido en `policy_v2.py`. Implementa
la interfaz `Policy` del framework:
- `mount()` carga la Q-table desde disco (cacheada para no releer cada juego).
- `act(s)` infiere el color, canonicaliza el tablero, aplica la heuristica de
  seguridad, y consulta la Q-table greedy. Tiempo: < 1 ms por movimiento.

## Cómo entrenar

### V2 (el agente del torneo)

```bash
# Desde la raiz del repo (donde esta connect4/)
python -m nico_agent.train_v2 --episodes 100000
```

Hiperparametros configurables:
- `--alpha 0.1`   tasa de aprendizaje TD
- `--c-ucb 1.41`  parametro de exploracion UCB1 (sqrt(2) por defecto)
- `--gamma 1.0`   factor de descuento (1.0 porque los episodios son cortos)
- `--no-symmetry` desactiva la augmentacion izq-der (para la ablacion)
- `--resume`      retoma desde data/qtable_v2.pkl si existe

Duracion estimada: ~3 minutos para 100k episodios en una maquina moderna.

### V1

```bash
python -m nico_agent.train_v1 --episodes 100000
```

## Cómo evaluar

```python
from nico_agent.policy_v0 import RandomPolicy
from nico_agent.policy_v2 import QLearningPolicy
from nico_agent.harness import evaluate

stats = evaluate(QLearningPolicy(), RandomPolicy(), n_games=1000, alternate_first=True)
print(f'Win rate: {stats.win_rate:.2%}')
print(f'  Como rojo:     {stats.as_red.win_rate:.2%}')
print(f'  Como amarillo: {stats.as_yellow.win_rate:.2%}')
```

## Cómo correr el notebook de análisis

```bash
cd nico_agent
jupyter notebook entrega.ipynb
```

El notebook genera 2 figuras en `data/exp*.png` que se usan en el PDF.

## Decisiones de diseño

**Canonicalización por jugador.** La Q-table indexa por `board * current_player`,
no por el tablero crudo. Asi mi color siempre es +1 y el del oponente -1,
independientemente de si estoy jugando rojo o amarillo. Esto reduce a la
mitad el espacio efectivo y hace que la actualizacion bipolar (Hoja 12) sea
natural: el valor del estado del oponente, visto desde mi perspectiva, es el
negativo de su Q.

**Actualizacion TD bipolar (V2).** En lugar de esperar al fin del episodio
(Monte Carlo de V1), actualizo Q en cada paso usando:

    Q(s,a) <- Q(s,a) + alpha * (r - gamma * max_a' Q(s', a') - Q(s,a))

El signo MENOS delante de gamma absorbe la bipolaridad: como s' esta
canonicalizado para el oponente, el max es desde *su* perspectiva, y para
mi vale lo opuesto. Esto es exactamente el AMG (Alternating Markov Games)
de la Hoja 12.

**UCB1 durante entrenamiento (V2).** En lugar de eps-greedy, uso UCB1
(Hoja 10) para seleccionar acciones durante self-play. El termino de
exploracion c * sqrt(ln N(s) / N(s,a)) garantiza que cada accion en cada
estado se visite con probabilidad creciente, lo que asegura convergencia
asintotica de Q a q* (teorema de convergencia de UCB1).

**Augmentacion con simetria izq-der.** Connect-4 tiene simetria horizontal:
el tablero espejado tiene el mismo valor estrategico. Aplico cada
actualizacion de Q tambien al estado-accion espejados (excepto cuando el
estado coincide consigo mismo). Esto duplica la experiencia efectiva sin
costo computacional adicional.

**Heuristica de seguridad activable.** Antes de consultar la Q-table, dos
reglas: (1) si tengo 4-en-linea inmediato, lo juego; (2) si el oponente
tiene 4-en-linea inmediato, lo bloqueo. Para estados nuevos no vistos en
entrenamiento, fallback a preferencia central (columna 3, luego 2/4, etc.)
en lugar de aleatorio. La heuristica se puede DESACTIVAR mediante
`QLearningPolicyNoHeuristic`, lo que permite el experimento de ablacion
en el notebook.

## Notas sobre el espacio de estados

Tras 100k episodios de self-play con simetria, la Q-table tiene del orden
de 10^6 estados unicos. El espacio teorico de tableros validos en Connect-4
es ~10^13, asi que cubrimos < 0.001% del total. Esto motiva la propuesta
de mejora futura: abstraccion de estados por features (amenazas 3-en-linea,
control del centro, alturas relativas) para lograr generalizacion.

## Referencias del curso

- **Hoja 10:** Bandits multi-armed, UCB1.
- **Hoja 11:** First-Visit Monte Carlo, eps-greedy.
- **Hoja 12:** MDPs competitivos (Alternating Markov Games), Q-learning bipolar.
