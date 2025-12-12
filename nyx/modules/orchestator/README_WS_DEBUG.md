# Debugging: ConnectionClosedOK en WebSockets

Este documento analiza el motivo de las excepciones `ConnectionClosedOK` en logs de "shielded future" y propone la solución definitiva.

## 1. El Problema

Al cerrar la aplicación (Ctrl+C) o cuando un cliente se desconecta, aparecen múltiples logs como este:

```text
ConnectionClosedOK exception in shielded future
websockets.exceptions.ConnectionClosedOK: received 1000 (OK); then sent 1000 (OK)
```

Esto indica que una conexión WebSocket se cerró correctamente (código 1000), pero la excepción utilizada por la librería `websockets` para el control de flujo (`ConnectionClosedOK`) fue capturada por `asyncio` dentro de una tarea protegida (`shielded`), sin ser manejada correctamente por el código de la aplicación.

## 2. Análisis de Causa Raíz

Tras revisar el código en `app/transport/receiver/loop.py` y `app/transport/sender/loop.py`, se ha detectado un **bug crítico** en la gestión de bucles.

### El Bug del Bucle Infinito

En la implementación actual de `ReceiveLoop` (y similar en `SenderLoop`), cuando se captura la desconexión, **el bucle no se rompe**.

```python
# CÓDIGO ACTUAL (Con Error)
while self._is_running:
    try:
        message = await self._websocket.receive()
        # ... procesar ...
    except (WebSocketDisconnect, ConnectionClosedOK):
        # Se notifica el cierre...
        await self._event_bus.publish(event="ws.close")

        # ERROR: ¡Falta un break!
        # El bucle continúa, vuelve a llamar a receive(),
        # y vuelve a lanzar ConnectionClosedOK inmediatamente.
```

Esto provoca que:

1.  Se entre en un bucle infinito de excepciones `ConnectionClosedOK`.
2.  Al ocurrir el shutdown (`Ctrl+C`), estas tareas están "girando" violentamente lanzando excepciones.
3.  El `event_bus` se satura de eventos `ws.close`.
4.  Eventualmente, durante la cancelación de tareas por parte de Uvicorn/FastAPI, una de estas excepciones se propaga a través de un futuro protegido (`shielded`), generando el log que ves.

## 3. Solución

La solución es **romper explícitamente el bucle** cuando se detecta el cierre de la conexión.

### Archivo: `app/transport/receiver/loop.py`

```python
# CORRECCIÓN EN ReceiveLoop._run
async def _run(self) -> None:
    try:
        while self._is_running:
            try:
                message:str = await self._websocket.receive()
                await self._event_bus.publish(event="ws.message.received", payload=message)

            except (WebSocketDisconnect, ConnectionClosedOK):
                # Notify closed connection.
                await self._event_bus.publish(event="ws.close")
                # SOLUCIÓN: Romper el bucle
                break  # <--- AÑADIR ESTE BREAK

            # ... resto del código ...
```

### Archivo: `app/transport/sender/loop.py`

```python
# CORRECCIÓN EN SenderLoop._run
async def _run(self) -> None:
    try:
        while self._is_running:
            try:
                # ... lógica de envío ...
                await self._websocket.send(message=message)

            except (WebSocketDisconnect, ConnectionClosedOK):
                # Notify closed connection.
                await self._event_bus.publish(event="ws.close")
                # SOLUCIÓN: Romper el bucle
                break  # <--- AÑADIR ESTE BREAK

            # ... resto del código ...
```

## 4. Resumen

El log "shielded future" es un síntoma de que una tarea en segundo plano terminó con una excepción no manejada durante el cierre. Al no romper el bucle `while` tras la desconexión, tus tareas `ReceiverLoop` y `SenderLoop` se convierten en generadores infinitos de excepciones `ConnectionClosedOK`. Añadiendo el `break` aseguras que la tarea finalice suavemente (return `None`), eliminando el ruido en los logs.
