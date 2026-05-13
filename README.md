# L.X.P-agent

Implementación de referencia de **L.X.P. (Latent Exchange Protocol)** y un agente IA
que lo utiliza con Groq.

L.X.P. cambia el patrón clásico de herramientas imperativas por estados semánticos:
los recursos se convierten en memoria latente, las capacidades se descubren como un
Hyper-Schema dinámico y los Ghost Workers anticipan resultados antes de que el modelo
termine de responder.

## Componentes

- **Zero-Hop Context (ZHC):** `VectorizedMemoryBridge` convierte archivos locales en
  recursos vectorizados, rankeados semánticamente para el prompt.
- **Ghost Workers:** ejecutores ligeros que anticipan cálculos seguros y sondas HTTP
  en paralelo.
- **Hyper-Schema Dinámico:** auto-descubre capacidades desde carpetas, archivos y URLs.
- **Cryptographic Intent Verification:** emite pruebas HMAC de intención para acciones
  pre-aprobadas; acciones sensibles como `write`, `delete` y `network` no están
  permitidas por defecto.
- **Agente Groq:** usa `meta-llama/llama-4-scout-17b-16e-instruct` por defecto.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Configura tu API key sin guardarla en git:

```bash
export GROQ_API_KEY="gsk_..."
```

## Uso

Mapear el repo como Zero-Hop Context y preguntar al agente:

```bash
lxp-agent "Explica el protocolo L.X.P y calcula: 21 * 2" --scan .
```

Usar un modelo explícito:

```bash
lxp-agent "Resume este proyecto" \
  --scan README.md \
  --model meta-llama/llama-4-scout-17b-16e-instruct
```

## Pruebas

```bash
ruff check .
pytest
```

## Nota de seguridad

El archivo `.env.example` muestra los nombres de variables esperados, pero las claves
reales deben mantenerse en variables de entorno o en un gestor de secretos.