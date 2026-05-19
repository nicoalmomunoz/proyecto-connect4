Agente MCTS - Juan Montes Sabogal

Qué hace el agente ahora:
- El agente usa Monte Carlo Tree Search (MCTS) para elegir una columna en Connect4.
- Funciona en ciclos: explora acciones posibles, simula partidas aleatorias desde esas acciones y usa los resultados para decidir.
- En la fase de selección usa UCT (una fórmula que equilibra jugar acciones con muchas victorias y explorar acciones menos probadas).
- Si en un estado no hay estadísticas todavía, el agente expande una acción nueva y continúa con simulaciones hasta el final de la partida.
- Al terminar las simulaciones acumula visitas y éxitos por acción y elige la mejor acción según esas estadísticas.

Cómo se comporta en la práctica:
- Tiene un límite de tiempo por movimiento y un tope de iteraciones para mantener rapidez en las pruebas.
- Si no alcanza a obtener estadísticas, elige una columna válida al azar.

Qué voy a cambiar antes del 26 de mayo:
- Aumentar el tiempo de búsqueda por movimiento para que el agente pueda explorar más y jugar con más fuerza.
- Fijar la semilla del generador aleatorio en las pruebas para que las corridas sean reproducibles.
- Añadir pruebas básicas que verifiquen que `act` devuelve siempre una columna válida y que el agente no rompe el entorno.
- Implementar una heurística simple en los rollouts: si hay una jugada que gana al instante o que evita una derrota inmediata, preferirla.
- Correr partidos contra el agente aleatorio y guardar un breve registro de resultados para la entrega.

Objetivo para el 26/05: entregar un agente más robusto, reproducible y con pruebas mínimas que demuestren su funcionamiento.