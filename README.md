# Agente Connect-4 — Minimax con evaluación heurística

**Autor:** Nicolas Almonacid Muñoz — Fundamentos de Inteligencia Artificial 2026.1
**Clase del agente:** `QLearningBipolarV2`

## Idea

Agente basado en **búsqueda adversarial pura**, sin pesos pre-entrenados ni
archivos de datos. Todo el conocimiento del juego está en código:

1. **Reglas reactivas de seguridad**: WIN inmediato y BLOCK al oponente.
2. **Minimax con poda alfa-beta** sobre el espacio de jugadas.
3. **Iterative deepening con presupuesto de tiempo**: empieza en
   profundidad 4 y va subiendo hasta agotar el tiempo. Siempre tiene
   una jugada lista en cualquier momento.
4. **Tabla de transposición en memoria**: cachea evaluaciones de
   subárboles para no recomputarlos. Posiciones repetidas (alcanzables
   por distintos órdenes de jugadas) se reusan, y la mejor jugada
   cacheada se prueba primero para podar más agresivamente.
5. **Función de evaluación heurística** basada en patrones de Connect-4:
   - Bonus por control del centro.
   - Conteo de "ventanas" de 4 casillas en las 4 direcciones
     (horizontal, vertical, diagonales), con score exponencial según
     cuántas fichas mías o del oponente contiene.
   - Sesgo defensivo (bloquear amenazas pesa un poco más que crear las
     propias).

## Cómo ejecutar

El archivo `policy.py` es **autocontenido**. Solo requiere `numpy` y la
clase `Policy` del framework `connect4`. No hay archivos de datos, no
hay paths, no hay dependencias extra.

```bash
# Estructura esperada:
#   <carpeta>/policy.py    (este agente)
#   connect4/              (framework del profesor)

python -c "from policy import QLearningBipolarV2; p = QLearningBipolarV2(); p.mount(); print('Listo')"
```

`mount()` acepta cualquier firma (con o sin timeout). Si recibe un
timeout, usa el 80% como presupuesto por movimiento. Default 1 segundo.

## Performance

Validación con 60 juegos vs `RandomPolicy` (30 como rojo, 30 como
amarillo), presupuesto de **solo 50 ms por movimiento**:

| Color | Wins | Tiempo medio/juego | Tiempo máx/juego |
|---|---|---|---|
| Rojo (primero) | **30/30 (100%)** | ~210 ms | 527 ms |
| Amarillo (segundo) | **30/30 (100%)** | ~260 ms | <600 ms |

Con presupuesto de 1 segundo por movimiento (default), la búsqueda
alcanza profundidad mayor a 10 plies y la TT acumula varios miles de
posiciones por turno.

Tiempo total por juego: <1% del minuto disponible.

## Decisiones de diseño (resumen)

- **Iterative deepening**: garantiza tener siempre una respuesta. Si el
  tiempo se acaba en mitad de una iteración profunda, la jugada de la
  iteración anterior (menos profunda pero completa) sigue siendo válida.
- **Move ordering inteligente**: la TT recuerda la mejor jugada de la
  iteración anterior; probarla primero hace que alfa-beta pode más.
- **Atajo terminal en cada nodo**: si alguna jugada cierra 4-en-línea,
  se devuelve inmediatamente sin explorar el resto.
- **Función de evaluación heurística estándar de Connect-4**: cuenta
  ventanas de 4 casillas con pesos exponenciales (4 fichas = victoria,
  3 fichas = amenaza, 2 fichas = setup, etc.). El bonus de centro
  refleja que las columnas centrales participan en más combinaciones
  ganadoras.
- **Sesgo defensivo**: en el score de ventanas, una amenaza del oponente
  pesa más (en valor absoluto) que crear la mía propia. Esto hace que
  el agente prefiera bloquear antes que avanzar agresivamente cuando
  hay duda.

## Por qué este enfoque

Connect-4 es un juego con espacio de estados grande (~10^13) pero
*acotado* y *adversarial*. Las técnicas clásicas de búsqueda con
evaluación heurística son altamente efectivas porque:

- El factor de ramificación es bajo (máximo 7 jugadas por turno).
- Las amenazas son locales y detectables con patrones simples.
- La poda alfa-beta multiplica enormemente la profundidad alcanzable.

Por eso un agente bien hecho con minimax + heurística + iterative
deepening puede competir con (e incluso superar a) sistemas más
sofisticados que dependen de modelos pre-entrenados.

## Referencias del curso

- **Hoja 4**: búsqueda en grafos.
- **Hoja 10**: criterios de exploración/explotación (UCB), aplicables al
  ordenamiento de jugadas en minimax.
- **Hoja 12**: MDPs competitivos (juegos alternantes), origen conceptual
  del minimax adversarial.
