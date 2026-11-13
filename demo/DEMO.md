# Demostración en vivo: guion

`uv run qae-demo` recorre el proyecto en 7 pasos y se detiene después de cada uno hasta que
pulses Enter. Todos los números sin ruido se calculan en vivo con el código de este
repositorio, unos 10 s de cómputo en total en un portátil. Los números del estudio completo
(E2–E4 tardan ~1.5 h en regenerarse) se leen de los archivos versionados en
`results/summary/`. Las figuras se guardan en `demo/out/` y se abren en el visor de imágenes
del sistema. La demo está en español; las figuras del estudio completo (`results/figures/`,
pasos 5 y 6) siguen en inglés.

## Antes de la charla (el día anterior, y otra vez 10 minutos antes)

```bash
cd qcc-project
git pull && uv sync                       # solo necesita red la primera vez
uv run qae-demo --no-pause --no-open      # ensayo: debe terminar en ~10-15 s sin errores
uv run pytest -m "not slow" -q            # opcional: 40 passed
```

- Terminal: letra de 18–20 pt y ventana de al menos **100 columnas** (las tablas caben en 80).
  Usa un tema claro si el proyector se ve deslavado.
- Cierra las ventanas del visor de imágenes del ensayo, para que cada figura se abra de nuevo
  durante la charla.
- Desactiva las notificaciones. Ten este archivo abierto en otra pantalla o impreso.
- **Plan B sin conexión:** tras un `uv sync`, la demo no necesita red. Si el portátil no
  puede ejecutarla, las figuras en `results/figures/` y las diapositivas cubren cada paso.

## Dónde va la demo en la charla

La presentación tiene la demo en dos partes:

| Diapositiva | Comando | Pasos |
|---|---|---|
| 9 · «Valorarla en vivo» | `uv run qae-demo --last 4` | 1–4: problema, operador A, Grover, IQAE frente a Monte Carlo (~6 min) |
| 14 · E4 ruido (opcional) | `uv run qae-demo --only 6` | 6: los mismos circuitos con el ruido de IBM (~1 min) |

Los pasos 5 (estudio completo) y 7 (conclusiones) repiten lo que muestran las diapositivas.
Están ahí para que `uv run qae-demo` funcione también como recorrido independiente, por
ejemplo si falla el proyector.

## Guion paso a paso (~8 minutos para los 7)

| # | Qué decir | Qué señalar en la salida | Tiempo |
|---|---|---|---|
| — | `uv run qae-demo` | Línea de título | — |
| 1 | «Un evento binario antes del vencimiento: 40 % malas noticias cerca de 85, 60 % buenas noticias cerca de 115. Un call digital paga 1 por encima de 100.» | Precio continuo **0.5636**; precio en la malla 0.5572 con n = 5; la figura muestra los dos modos y la región de pago en naranja. | 1 min |
| 2 | «El precio se convierte en la probabilidad de medir 1 en un qubit. A = cargador + comparador; un uso de A es una *consulta al oráculo*.» | El circuito (`P(S)` y luego `i ≥ K`); la probabilidad del vector de estado coincide con la cola de la malla en 10 dígitos; **2,114 compuertas de dos qubits por consulta** en hardware. | 1 min |
| 3 | «El operador de Grover convierte la amplitud en un ángulo de rotación. Tras k pasos la probabilidad es sin²((2k+1)θ). Leer una frecuencia requiere ~1/ε consultas; promediar lanzamientos de moneda, ~1/ε².» | La columna del circuito es exactamente la de la fórmula; las muestras de 100 shots se dispersan alrededor. La figura muestra la oscilación. | 1.5 min |
| 4 | «Una corrida de IQAE, ronda por ronda: elige potencias de Grover mayores a medida que el intervalo se reduce.» Luego: «Ahora una carrera: 20 semillas, y Monte Carlo recibe exactamente el mismo presupuesto.» | Rondas k = 0, 1, 1, 11, 51, 262; error final ~10⁻⁵. En la tabla de la carrera, la columna **MC / IQAE** crece a medida que baja el error objetivo (≈ 2×, 18×, 250×, 1,100×). Ese crecimiento es la aceleración cuadrática. Figura: azul con pendiente −1, naranja con pendiente −½. | 2 min |
| 5 | «Eso fue una muestra en vivo. El estudio completo son 49 configuraciones × 50 semillas.» | Pendientes −1.01 (IQAE) frente a −0.52 (MC); **3,659 frente a 117,102** consultas con ε = 10⁻³; luego la salvedad: **quasi-MC necesita 172**. | 1 min |
| 6 | «Ahora los mismos circuitos con el modelo de ruido de IBM.» | Con escala 1 la desviación salta de 0.04 (k = 0) a 0.3–0.5 para k ≥ 1: P(1) colapsa a ~0.5. Con el 3 % del ruido actual sobrevive hasta k ≈ 4. | 1 min |
| 7 | «Cinco conclusiones.» | Leer el panel; volver a las diapositivas para las conclusiones y las preguntas. | 0.5 min |

Los números en vivo del paso 4 y las probabilidades con ruido del paso 6 dependen de la
semilla. Son reproducibles con la semilla por defecto `--seed 2026`, así que los números de
arriba son los que verás.

## Si algo sale mal

| Problema | Solución |
|---|---|
| Un paso falla o se cuelga | Ctrl-C salta el paso; o reinicia en el siguiente con `uv run qae-demo --step N` |
| Hay que repetir un paso (una pregunta del público) | `uv run qae-demo --only 3` |
| Vas tarde | `uv run qae-demo --step 4 --quick` (carrera + estudio + ruido + conclusiones, ~5 s) |
| Las figuras no se abren | Están en `demo/out/` y `results/figures/`; ábrelas a mano, o vuelve a ejecutar con `--no-open` y usa las diapositivas |
| `uv` no está instalado en el equipo de la presentación | Instálalo (README, Quickstart) y luego `uv sync`. Necesita red, así que hazlo antes |

## Preguntas probables

- **«Entonces, ¿hay ventaja cuántica?»** Solo en número de consultas al oráculo frente a
  Monte Carlo simple, y solo sin ruido. QMC gana en este problema unidimensional, y el ruido
  la elimina por completo.
- **«¿Por qué un pago digital?»** No tiene error de aproximación del pago, así que todo error
  medido es error de estimación. En el call europeo, el factor de reescalamiento c
  intercambia error de aproximación por error de estimación amplificado (E1).
- **«¿Por qué la simulación es tan rápida?»** El sampler calcula una sola vez la distribución
  de salida exacta de cada circuito, a partir de unitarias en caché (Q^k es una potencia de
  matriz), y luego toma muestras multinomiales con semilla. Las pruebas lo comparan con la
  simulación compuerta a compuerta de Aer hasta 10⁻¹⁰.
- **«¿Qué haría falta en hardware real?»** IQAE con ε = 10⁻³ necesita ~55,000 compuertas de
  dos qubits en una línea. Con ruido realista solo se pueden usar ~344 (k = 1) (E3, E4).
- **«¿Es el cargador el cuello de botella?»** Aquí no: el comparador es el 98 % de las
  compuertas de dos qubits de A con n = 5. Aun así, el cargador crece como 2ⁿ (E5).
