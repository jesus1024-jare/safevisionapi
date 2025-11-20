from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
import os
import time
import threading
from queue import Queue
from typing import Optional, List
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import signal
import sys

# Inicializar FastAPI
app = FastAPI(title="API Detección YOLO ESP32", version="1.0.0")

# Configurar CORS para permitir conexiones desde el ESP32
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo YOLO
MODEL_PATH = "best.pt"
model = None

def load_model():
    """Carga el modelo YOLO"""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"✅ Modelo YOLO cargado desde {MODEL_PATH}")

# Cargar modelo al iniciar
load_model()

# Mostrar clases disponibles en el modelo (para debugging)
if model is not None:
    print(f"📋 Clases disponibles en el modelo: {list(model.names.values())}")

# Configuración global
CONFIDENCE_THRESHOLD = 0.25
active_streams = {}
stream_stats = {}
last_successful_camera_url = None  # Guardar la última URL exitosa
active_mjpeg_stream = False  # Flag para saber si hay un stream MJPEG activo
stream_should_stop = False  # Flag para detener el stream manualmente
last_status_check_time = 0  # Timestamp de la última verificación de status
cached_connection_status = 0  # Estado de conexión cacheado
STATUS_CACHE_DURATION = 5.0  # Segundos que dura el cache del status

# Configuración de captura automática
CAPTURE_FOLDER = "captures"  # Carpeta donde se guardan las capturas
CAPTURE_CLASSES = ["casco", "guantes", "chaleco"]  # Clases que activan la captura
CAPTURE_COOLDOWN = 5.0  # Segundos entre capturas (evita demasiadas capturas)
last_capture_time = {}  # Último tiempo de captura por cliente/stream
capture_count = 0  # Contador global de capturas

# Variables para captura automática desde el stream
last_stream_frame = None  # Último frame del stream
last_stream_frame_lock = threading.Lock()  # Lock para acceso thread-safe
last_stream_frame_time = 0  # Timestamp del último frame
auto_capture_thread = None  # Thread para capturas automáticas
auto_capture_running = False  # Flag para controlar el thread de captura

# Queue y thread para guardado asíncrono de capturas (evita bloquear el stream)
capture_queue = Queue(maxsize=10)  # Queue con límite para evitar acumulación excesiva
capture_worker_thread = None  # Thread worker para procesar capturas
capture_worker_running = False  # Flag para controlar el worker

# Crear carpeta de capturas si no existe
os.makedirs(CAPTURE_FOLDER, exist_ok=True)
print(f"📁 Carpeta de capturas: {os.path.abspath(CAPTURE_FOLDER)}")

# Lock para hacer thread-safe la obtención del siguiente número de captura
capture_number_lock = threading.Lock()

def get_next_capture_number() -> int:
    """
    Obtiene el siguiente número disponible para el formato capture_detect{N}
    Thread-safe para evitar colisiones cuando múltiples threads guardan simultáneamente
    
    Returns:
        El siguiente número disponible (ej: 0, 1, 2, ...)
    """
    global capture_number_lock
    
    with capture_number_lock:
        if not os.path.exists(CAPTURE_FOLDER):
            return 0
        
        # Buscar todos los archivos que empiecen con "capture_detect"
        existing_numbers = []
        for filename in os.listdir(CAPTURE_FOLDER):
            if filename.startswith("capture_detect") and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Ignorar archivos anotados
                if filename.endswith("_annotated.jpg") or filename.endswith("_annotated.jpeg") or filename.endswith("_annotated.png"):
                    continue
                # Extraer el número del nombre (capture_detect{N}.jpg)
                try:
                    # Remover extensión y prefijo
                    name_without_ext = os.path.splitext(filename)[0]  # capture_detect{N}
                    number_str = name_without_ext.replace("capture_detect", "")
                    if number_str.isdigit():
                        existing_numbers.append(int(number_str))
                except:
                    continue
        
        # Si no hay archivos, empezar desde 0
        if not existing_numbers:
            return 0
        
        # Retornar el siguiente número disponible
        return max(existing_numbers) + 1

# Modelos Pydantic para requests
class DetectionRequest(BaseModel):
    confidence: Optional[float] = 0.25

class FrameRequest(BaseModel):
    frame_base64: str
    confidence: Optional[float] = 0.25

def save_capture(frame: np.ndarray, detections: List[dict], source: str = "unknown") -> Optional[str]:
    """
    Guarda una captura cuando se detectan las clases especificadas
    
    Args:
        frame: Frame de video (numpy array)
        detections: Lista de detecciones
        source: Origen de la captura (para identificar el stream)
    
    Returns:
        Ruta del archivo guardado o None si no se guardó
    """
    global capture_count, last_capture_time
    
    # Verificar si hay alguna de las clases objetivo en las detecciones
    detected_classes = [det["class"].lower() for det in detections]
    target_detected = any(target_class.lower() in detected_classes for target_class in CAPTURE_CLASSES)
    
    if not target_detected:
        return None
    
    # Verificar cooldown (evitar demasiadas capturas)
    current_time = time.time()
    if source in last_capture_time:
        time_since_last = current_time - last_capture_time[source]
        if time_since_last < CAPTURE_COOLDOWN:
            return None  # Aún en cooldown
    
    try:
        # Obtener el siguiente número disponible
        capture_number = get_next_capture_number()
        filename = f"capture_detect{capture_number}.jpg"
        filepath = os.path.join(CAPTURE_FOLDER, filename)
        
        # Guardar frame ORIGINAL sin bounding boxes
        # Los bounding boxes se dibujarán cuando se analice la captura
        # Esto permite que el modelo YOLO pueda detectar correctamente al analizar
        cv2.imwrite(filepath, frame)
        
        # Actualizar contadores
        capture_count += 1
        last_capture_time[source] = current_time
        
        print(f"📸 Captura guardada: {filename} (Total: {capture_count})")
        return filepath
        
    except Exception as e:
        print(f"❌ Error guardando captura: {str(e)}")
        return None

def capture_worker():
    """
    Worker thread que procesa las capturas de la queue de forma continua
    Este thread se ejecuta de forma independiente y no bloquea el stream principal
    """
    global capture_count, last_capture_time, capture_worker_running
    
    while capture_worker_running:
        try:
            # Obtener una captura de la queue (bloquea hasta que haya una disponible o timeout)
            try:
                item = capture_queue.get(timeout=1.0)
            except:
                continue  # Timeout, verificar si sigue corriendo
            
            if item is None:  # Señal para detener
                break
            
            frame_copy, detections, source, current_time = item
            
            try:
                # VERIFICACIONES: Hacer todas las verificaciones en el worker thread, no en el principal
                # Verificar si hay alguna de las clases objetivo en las detecciones
                detected_classes = [det["class"].lower() for det in detections]
                target_detected = any(target_class.lower() in detected_classes for target_class in CAPTURE_CLASSES)
                
                if not target_detected:
                    # No hay clases objetivo, no guardar
                    # task_done() se llamará en el finally
                    continue
                
                # Verificar cooldown (evitar demasiadas capturas)
                if source in last_capture_time:
                    time_since_last = current_time - last_capture_time[source]
                    if time_since_last < CAPTURE_COOLDOWN:
                        # Aún en cooldown, no guardar
                        # task_done() se llamará en el finally
                        continue
                
                # Obtener el siguiente número disponible
                capture_number = get_next_capture_number()
                filename = f"capture_detect{capture_number}.jpg"
                filepath = os.path.join(CAPTURE_FOLDER, filename)
                
                # Guardar frame ORIGINAL sin bounding boxes
                # Los bounding boxes se dibujarán cuando se analice la captura
                # Esto permite que el modelo YOLO pueda detectar correctamente al analizar
                cv2.imwrite(filepath, frame_copy)
                
                # Actualizar contadores
                capture_count += 1
                last_capture_time[source] = current_time
                
                print(f"📸 Captura guardada: {filename} (Total: {capture_count})")
                
            except Exception as e:
                print(f"❌ Error procesando captura en worker: {str(e)}")
            finally:
                # Marcar la tarea como completada
                capture_queue.task_done()
                
        except Exception as e:
            print(f"❌ Error en worker de capturas: {str(e)}")
            time.sleep(0.1)  # Esperar un poco antes de reintentar

def _copy_and_queue_frame(frame: np.ndarray, detections: List[dict], source: str, current_time: float):
    """
    Función auxiliar que se ejecuta en un thread separado para copiar el frame y encolarlo.
    Esta función NO debe ser llamada directamente desde el thread principal del stream.
    """
    try:
        # Verificar si hay espacio en la queue
        if capture_queue.full():
            return
        
        # Copiar frame (esta operación puede tomar tiempo, por eso se hace en thread separado)
        frame_copy = frame.copy()
        
        # Agregar a la queue sin bloqueo
        capture_queue.put_nowait((frame_copy, detections, source, current_time))
    except:
        # Cualquier error se ignora completamente
        pass

def save_capture_async(frame: np.ndarray, detections: List[dict], source: str = "unknown"):
    """
    Agrega una captura a la queue para ser procesada de forma asíncrona
    Esta función es COMPLETAMENTE NO-BLOQUEANTE y retorna inmediatamente
    NO hace NINGUNA verificación ni procesamiento en el thread principal
    TODO el trabajo se hace en threads separados
    
    IMPORTANTE: Esta función NO debe bloquear el stream bajo NINGUNA circunstancia.
    La copia del frame se hace en un thread separado para garantizar que el stream
    continúe inmediatamente sin ninguna pausa.
    
    Args:
        frame: Frame de video (numpy array) - se hace una copia en thread separado
        detections: Lista de detecciones
        source: Origen de la captura (para identificar el stream)
    """
    # SOLUCIÓN ÓPTIMA: Lanzar un thread daemon que hace la copia y encola
    # El thread principal del stream retorna INMEDIATAMENTE sin esperar nada
    # Esto garantiza que el stream nunca se detenga
    try:
        # Verificar rápidamente si hay espacio (operación muy rápida)
        if capture_queue.full():
            return
        
        # Lanzar thread daemon que hace la copia y encola (fire and forget)
        # El thread principal retorna inmediatamente sin esperar
        copy_thread = threading.Thread(
            target=_copy_and_queue_frame,
            args=(frame, detections, source, time.time()),
            daemon=True
        )
        copy_thread.start()
        # NO esperamos a que termine - el stream continúa inmediatamente
    except:
        # Cualquier error se ignora completamente
        # El stream DEBE continuar sin importar qué pase
        pass

def save_auto_capture(frame: np.ndarray, source: str = "auto_capture") -> Optional[str]:
    """
    Guarda una captura automática SIN bounding boxes (frame original)
    Los bounding boxes se dibujarán cuando se analice la captura
    
    Args:
        frame: Frame de video (numpy array) - frame original sin anotaciones
        source: Origen de la captura
    
    Returns:
        Ruta del archivo guardado o None si no se guardó
    """
    global capture_count
    
    try:
        # Obtener el siguiente número disponible
        capture_number = get_next_capture_number()
        filename = f"capture_detect{capture_number}.jpg"
        filepath = os.path.join(CAPTURE_FOLDER, filename)
        
        # Guardar frame ORIGINAL sin bounding boxes
        # Esto permite que cuando se analice la captura, el modelo YOLO pueda detectar correctamente
        cv2.imwrite(filepath, frame)
        
        # Actualizar contador
        capture_count += 1
        
        print(f"📸 Captura automática guardada (sin bounding boxes): {filename} (Total: {capture_count})")
        return filepath
        
    except Exception as e:
        print(f"❌ Error guardando captura automática: {str(e)}")
        return None

def auto_capture_worker():
    """
    Worker thread que captura frames automáticamente desde el stream activo
    """
    global last_stream_frame, last_stream_frame_lock, last_stream_frame_time, auto_capture_running, active_mjpeg_stream
    
    while auto_capture_running:
        try:
            # Determinar intervalo según si hay stream activo
            if active_mjpeg_stream:
                interval = 5.0  # 5 segundos si el stream está activo (aumentado de 3.0)
            else:
                interval = 7.0  # 7 segundos si no hay stream activo (aumentado de 5.0)
            
            time.sleep(interval)
            
            if not auto_capture_running:
                break
            
            # Obtener el último frame del stream de forma thread-safe
            with last_stream_frame_lock:
                if last_stream_frame is not None:
                    # Hacer una copia del frame para evitar problemas de concurrencia
                    frame_copy = last_stream_frame.copy()
                    frame_time = last_stream_frame_time
                else:
                    frame_copy = None
            
            # Si tenemos un frame válido (menos de 10 segundos de antigüedad)
            if frame_copy is not None and (time.time() - frame_time) < 10.0:
                save_auto_capture(frame_copy, source="stream_auto")
            else:
                # No hay frame disponible o es muy antiguo
                if not active_mjpeg_stream:
                    print("⏸️  No hay stream activo, esperando...")
                
        except Exception as e:
            print(f"❌ Error en worker de captura automática: {str(e)}")
            time.sleep(1.0)  # Esperar un poco antes de reintentar

# Endpoints

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Detección YOLO para ESP32",
        "version": "1.0.0",
        "endpoints": {
            "/": "Información de la API",
            "/health": "Estado de salud del servicio",
            "/detect": "POST - Detectar objetos en una imagen (base64)",
            "/stream": "WebSocket - Stream de video en tiempo real",
            "/stream/video": "GET - Stream MJPEG con detecciones",
            "/camera/status": "GET - Estado de conexión con la cámara (1=conectado, 0=desconectado)",
            "/stats": "GET - Estadísticas del stream activo",
            "/config": "GET/POST - Configuración del modelo",
            "/captures": "GET - Listar capturas guardadas",
            "/captures/{filename}": "GET - Obtener una imagen de captura específica"
        }
    }

@app.get("/health")
async def health():
    """Verificar estado de salud del servicio"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "active_streams": len(active_streams)
    }

@app.get("/config")
async def get_config():
    """Obtener configuración actual"""
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_path": MODEL_PATH,
        "model_loaded": model is not None,
        "capture_classes": CAPTURE_CLASSES,
        "capture_cooldown": CAPTURE_COOLDOWN,
        "capture_folder": os.path.abspath(CAPTURE_FOLDER),
        "total_captures": capture_count
    }

@app.post("/config")
async def set_config(request: DetectionRequest):
    """Actualizar configuración"""
    global CONFIDENCE_THRESHOLD
    if request.confidence is not None:
        if 0.0 <= request.confidence <= 1.0:
            CONFIDENCE_THRESHOLD = request.confidence
            return {"message": "Configuración actualizada", "confidence_threshold": CONFIDENCE_THRESHOLD}
        else:
            raise HTTPException(status_code=400, detail="Confianza debe estar entre 0.0 y 1.0")
    return {"message": "No se proporcionaron cambios"}

@app.post("/detect")
async def detect_objects(request: FrameRequest):
    """
    Detectar objetos en una imagen enviada como base64
    El ESP32 puede enviar frames individuales aquí
    """
    try:
        # Decodificar imagen base64
        image_data = base64.b64decode(request.frame_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
        
        # Realizar detección
        confidence = request.confidence if request.confidence else CONFIDENCE_THRESHOLD
        results = model(frame, conf=confidence, verbose=False)
        
        # Procesar resultados
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = results[0].names[cls]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })
        
        # Guardar captura si se detectan las clases objetivo
        capture_path = save_capture(frame, detections, source="detect_endpoint")
        capture_saved = capture_path is not None
        
        response = {
            "detections": detections,
            "count": len(detections),
            "timestamp": time.time(),
            "capture_saved": capture_saved
        }
        
        if capture_saved:
            response["capture_path"] = capture_path
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket para stream de video en tiempo real desde ESP32
    El ESP32 envía frames y recibe resultados de detección
    """
    await websocket.accept()
    client_id = f"client_{int(time.time())}"
    active_streams[client_id] = {
        "websocket": websocket,
        "start_time": time.time(),
        "frame_count": 0,
        "detections_count": 0,
        "fps": 0
    }
    
    print(f"✅ Cliente conectado: {client_id}")
    
    try:
        fps_start_time = time.time()
        fps_counter = 0
        
        while True:
            # Recibir frame del ESP32
            data = await websocket.receive_json()
            
            if "frame" not in data:
                await websocket.send_json({"error": "No se recibió frame"})
                continue
            
            # Decodificar frame base64
            try:
                frame_data = base64.b64decode(data["frame"])
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Frame inválido"})
                    continue
            except Exception as e:
                await websocket.send_json({"error": f"Error decodificando frame: {str(e)}"})
                continue
            
            # Obtener confianza del request o usar la global
            confidence = data.get("confidence", CONFIDENCE_THRESHOLD)
            
            # Realizar detección
            start_detection = time.time()
            results = model(frame, conf=confidence, verbose=False)
            detection_time = time.time() - start_detection
            
            # Procesar resultados
            detections = []
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results[0].names[cls]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                
                detections.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })
            
            # Guardar captura si se detectan las clases objetivo (asíncrono para no bloquear el stream)
            # Verificar si se debe guardar antes de iniciar el thread
            detected_classes = [det["class"].lower() for det in detections]
            target_detected = any(target_class.lower() in detected_classes for target_class in CAPTURE_CLASSES)
            capture_saved = False
            if target_detected:
                # Verificar cooldown
                current_time = time.time()
                source_key = f"websocket_{client_id}"
                if source_key not in last_capture_time or (current_time - last_capture_time[source_key]) >= CAPTURE_COOLDOWN:
                    save_capture_async(frame, detections, source=source_key)
                    capture_saved = True
            
            # Calcular FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
                active_streams[client_id]["fps"] = current_fps
            else:
                current_fps = active_streams[client_id]["fps"]
            
            # Actualizar estadísticas
            active_streams[client_id]["frame_count"] += 1
            active_streams[client_id]["detections_count"] += len(detections)
            
            # Enviar respuesta al ESP32
            response = {
                "detections": detections,
                "count": len(detections),
                "fps": current_fps,
                "detection_time_ms": round(detection_time * 1000, 2),
                "timestamp": time.time(),
                "capture_saved": capture_saved
            }
            
            # Nota: capture_path no está disponible aquí porque save_capture_async es asíncrono
            # y no retorna el path inmediatamente
            
            await websocket.send_json(response)
    
    except WebSocketDisconnect:
        print(f"❌ Cliente desconectado: {client_id}")
    except Exception as e:
        print(f"❌ Error en WebSocket: {str(e)}")
    finally:
        if client_id in active_streams:
            del active_streams[client_id]

@app.get("/stream/video")
async def stream_video_with_detections():
    """
    Endpoint para streaming de video con detecciones (MJPEG stream)
    El ESP32 puede conectarse aquí para recibir video procesado
    """
    def generate():
        global last_successful_camera_url, active_mjpeg_stream, auto_capture_thread, auto_capture_running, last_stream_frame, last_stream_frame_lock, last_stream_frame_time, stream_should_stop
        
        active_mjpeg_stream = True  # Marcar que hay un stream activo
        stream_should_stop = False  # Resetear flag al iniciar
        # El thread de captura automática ya está corriendo desde el inicio del programa
        
        # URLs alternativas a probar (ordenadas por probabilidad de éxito)
        base_urls = [
            "http://192.168.0.102/stream",  # URL base (menos probable que funcione)
        ]
        
        # Si tenemos una URL exitosa previa, probarla primero
        if last_successful_camera_url:
            if last_successful_camera_url in base_urls:
                # Mover la URL exitosa al principio
                base_urls.remove(last_successful_camera_url)
            alt_urls = [last_successful_camera_url] + base_urls
        else:
            alt_urls = base_urls
        
        cap = None
        camera_url = None
        
        # Intentar conectar a cada URL
        connection_errors = []
        for url in alt_urls:
            try:
                print(f"Intentando conectar a: {url}")
                cap = cv2.VideoCapture(url)
                # Configurar timeout para evitar bloqueos
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Configurar timeout de conexión (en milisegundos) - 3 segundos
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                
                if cap.isOpened():
                    # Intentar leer un frame con timeout usando threading
                    # para evitar esperar 30 segundos si el stream no responde
                    frame_read = [False, None, None]
                    
                    def read_frame():
                        try:
                            # Intentar leer frame - esto puede fallar con errores de FFmpeg
                            ret, frame = cap.read()
                            frame_read[0] = True
                            frame_read[1] = ret
                            frame_read[2] = frame
                        except SystemExit:
                            # No capturar SystemExit
                            raise
                        except KeyboardInterrupt:
                            # No capturar KeyboardInterrupt
                            raise
                        except Exception as e:
                            # Capturar CUALQUIER otro error (incluyendo errores de FFmpeg/assertion)
                            # Los errores de assertion de FFmpeg se capturan aquí
                            frame_read[0] = True
                            frame_read[1] = False
                            frame_read[2] = None
                    
                    read_thread = threading.Thread(target=read_frame, daemon=True)
                    read_thread.start()
                    read_thread.join(timeout=3.0)  # Timeout de 3 segundos para leer frame
                    
                    if not frame_read[0]:
                        # El thread no terminó, el stream no responde
                        error_msg = f"Timeout: El stream en {url} no responde (más de 3 segundos)"
                        connection_errors.append(error_msg)
                        print(f"⏱️  {error_msg}")
                        cap.release()
                        cap = None
                        continue
                    
                    ret, test_frame = frame_read[1], frame_read[2]
                    
                    if ret and test_frame is not None:
                        camera_url = url
                        last_successful_camera_url = url  # Guardar URL exitosa
                        print(f"✅ Conectado exitosamente a: {url}")
                        break
                    else:
                        error_msg = f"No se pudo leer frame desde {url} (stream vacío o inválido)"
                        connection_errors.append(error_msg)
                        print(f"⚠️  {error_msg}")
                        cap.release()
                        cap = None
                else:
                    error_msg = f"No se pudo abrir conexión a {url}"
                    connection_errors.append(error_msg)
                    print(f"⚠️  {error_msg}")
            except Exception as e:
                error_msg = f"Error al conectar a {url}: {str(e)}"
                connection_errors.append(error_msg)
                print(f"❌ {error_msg}")
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                    cap = None
        
        if not cap or not cap.isOpened() or camera_url is None:
            # Crear mensaje de error más detallado
            error_details = "\n".join(connection_errors[-3:])  # Mostrar últimos 3 errores
            error_msg = (
                '--frame\r\n'
                'Content-Type: text/plain\r\n\r\n'
                'Error: No se pudo conectar a la cámara en 192.168.0.103\r\n'
                '\r\n'
                'Posibles causas:\r\n'
                '1. El dispositivo no está encendido\r\n'
                '2. La IP es incorrecta (verifique en el router o configuración del dispositivo)\r\n'
                '3. El dispositivo está en una red diferente\r\n'
                '4. Hay un firewall bloqueando la conexión\r\n'
                '5. El dispositivo no tiene un stream MJPEG disponible\r\n'
                '\r\n'
                f'Errores: {error_details}\r\n'
                '\r\n'
            )
            yield error_msg.encode('utf-8')
            return
        
        frame_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5  # Reducido para reconectar más rápido
        reconnect_delay = 2.0  # Segundos entre reconexiones
        
        # Loop principal del stream - se mantiene activo indefinidamente
        # Este loop NUNCA debe terminar a menos que sea explícitamente detenido
        try:
            while True:
                # Verificar si se debe detener el stream manualmente
                if stream_should_stop:
                    print("🛑 Stream detenido manualmente por endpoint /stream/stop")
                    stream_should_stop = False  # Resetear flag
                    break
                
                try:
                    # Leer frame con timeout usando threading para evitar bloqueos largos
                    frame_read = [False, None, None]
                    
                    def read_frame():
                        try:
                            # Intentar leer frame - esto puede fallar con errores de FFmpeg
                            ret, frame = cap.read()
                            frame_read[0] = True
                            frame_read[1] = ret
                            frame_read[2] = frame
                        except SystemExit:
                            # No capturar SystemExit
                            raise
                        except KeyboardInterrupt:
                            # No capturar KeyboardInterrupt
                            raise
                        except Exception as e:
                            # Capturar CUALQUIER otro error (incluyendo errores de FFmpeg/assertion)
                            # Los errores de assertion de FFmpeg se capturan aquí
                            frame_read[0] = True
                            frame_read[1] = False
                            frame_read[2] = None
                    
                    read_thread = threading.Thread(target=read_frame, daemon=True)
                    read_thread.start()
                    read_thread.join(timeout=5.0)  # Timeout de 5 segundos para leer frame
                    
                    # Verificar si el thread terminó
                    if not frame_read[0]:
                        # Timeout al leer frame - el stream no responde
                        print(f"⏱️  Timeout leyendo frame. Intentando reconectar...")
                        consecutive_errors += 1
                        
                        # Cerrar conexión actual de forma segura
                        if cap:
                            try:
                                # Forzar cierre de la conexión para evitar errores de FFmpeg
                                cap.release()
                                # Dar tiempo para que se liberen los recursos
                                time.sleep(0.1)
                            except Exception as release_error:
                                # Ignorar errores al liberar, la conexión ya está rota
                                pass
                            finally:
                                cap = None
                        
                        # Intentar reconectar
                        if consecutive_errors < max_consecutive_errors:
                            time.sleep(reconnect_delay)
                            # Reconectar usando la misma URL exitosa
                            # IMPORTANTE: Usar un enfoque más seguro para evitar errores de FFmpeg
                            try:
                                print(f"🔄 Intentando reconectar a: {camera_url}")
                                
                                # Cerrar cualquier conexión anterior de forma segura
                                if cap:
                                    try:
                                        cap.release()
                                    except:
                                        pass
                                    cap = None
                                
                                # Esperar un poco más antes de reconectar para que FFmpeg se limpie
                                time.sleep(1.0)
                                
                                # Crear nueva conexión
                                new_cap = cv2.VideoCapture(camera_url)
                                new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                                
                                if new_cap.isOpened():
                                    # Verificar que podemos leer un frame con timeout más corto
                                    test_read = [False, None, None]
                                    
                                    def test_read_frame():
                                        try:
                                            # Intentar leer frame con validación adicional
                                            ret, frame = new_cap.read()
                                            # Validar que el frame es válido
                                            if ret and frame is not None and frame.size > 0:
                                                test_read[0] = True
                                                test_read[1] = True
                                                test_read[2] = frame
                                            else:
                                                test_read[0] = True
                                                test_read[1] = False
                                                test_read[2] = None
                                        except SystemExit:
                                            raise
                                        except KeyboardInterrupt:
                                            raise
                                        except Exception as e:
                                            # Capturar cualquier error de FFmpeg/OpenCV
                                            test_read[0] = True
                                            test_read[1] = False
                                            test_read[2] = None
                                    
                                    test_thread = threading.Thread(target=test_read_frame, daemon=True)
                                    test_thread.start()
                                    test_thread.join(timeout=2.0)  # Timeout más corto
                                    
                                    if test_read[0] and test_read[1] and test_read[2] is not None:
                                        cap = new_cap
                                        print(f"✅ Reconectado exitosamente")
                                        consecutive_errors = 0
                                        continue
                                    else:
                                        try:
                                            new_cap.release()
                                        except:
                                            pass
                            except Exception as e:
                                print(f"❌ Error al reconectar: {str(e)}")
                                # Esperar más tiempo antes de reintentar para evitar errores de FFmpeg
                                time.sleep(2.0)
                        
                        # Si no se pudo reconectar, esperar un poco más antes de reintentar
                        if not cap or not cap.isOpened():
                            time.sleep(reconnect_delay)
                            continue
                    
                    ret, frame = frame_read[1], frame_read[2]
                    
                    if not ret or frame is None:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"⚠️  Demasiados errores consecutivos ({consecutive_errors}). Intentando reconectar...")
                            consecutive_errors = 0  # Resetear para intentar reconectar
                            
                            # Cerrar y reconectar de forma segura
                            if cap:
                                try:
                                    # Forzar cierre para evitar errores de FFmpeg
                                    cap.release()
                                    time.sleep(0.1)  # Dar tiempo para liberar recursos
                                except Exception as release_error:
                                    # Ignorar errores al liberar
                                    pass
                                finally:
                                    cap = None
                            
                            time.sleep(reconnect_delay)
                            
                            # Intentar reconectar
                            try:
                                print(f"🔄 Reconectando a: {camera_url}")
                                new_cap = cv2.VideoCapture(camera_url)
                                new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                                
                                if new_cap.isOpened():
                                    # Verificar que podemos leer un frame
                                    test_read = [False, None, None]
                                    
                                    def test_read_frame():
                                        try:
                                            ret, frame = new_cap.read()
                                            test_read[0] = True
                                            test_read[1] = ret
                                            test_read[2] = frame
                                        except SystemExit:
                                            raise
                                        except KeyboardInterrupt:
                                            raise
                                        except Exception:
                                            # Capturar cualquier error de FFmpeg/OpenCV
                                            test_read[0] = True
                                            test_read[1] = False
                                            test_read[2] = None
                                    
                                    test_thread = threading.Thread(target=test_read_frame, daemon=True)
                                    test_thread.start()
                                    test_thread.join(timeout=3.0)
                                    
                                    if test_read[0] and test_read[1] and test_read[2] is not None:
                                        cap = new_cap
                                        print(f"✅ Reconectado")
                                        continue
                                    else:
                                        new_cap.release()
                            except Exception as e:
                                print(f"❌ Error al reconectar: {str(e)}")
                                time.sleep(reconnect_delay)
                                continue
                        
                        time.sleep(0.1)  # Esperar un poco antes de reintentar
                        continue
                    
                    # Frame leído exitosamente
                    consecutive_errors = 0
                    frame_count += 1
                    
                    # Guardar el último frame para captura automática (thread-safe)
                    # Hacer la copia en un thread separado para no bloquear el stream
                    def update_last_frame():
                        global last_stream_frame, last_stream_frame_time
                        try:
                            frame_copy = frame.copy()
                            with last_stream_frame_lock:
                                last_stream_frame = frame_copy
                                last_stream_frame_time = time.time()
                        except Exception:
                            pass
                    
                    # Lanzar thread daemon para actualizar el último frame (no bloquea)
                    update_thread = threading.Thread(target=update_last_frame, daemon=True)
                    update_thread.start()
                    
                    # Realizar detección
                    try:
                        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                        
                        # Procesar detecciones para captura
                        detections = []
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = results[0].names[cls]
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                            
                            detections.append({
                                "class": class_name,
                                "confidence": float(conf),
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2)
                                }
                            })
                        
                        # Dibujar detecciones PRIMERO (sin esperar nada)
                        # Esto asegura que el stream continúe sin interrupciones
                        annotated_frame = results[0].plot()
                        
                        # Guardar captura si se detectan las clases objetivo (asíncrono para no bloquear el stream)
                        # IMPORTANTE: El guardado se hace DESPUÉS de preparar el frame para el stream
                        # La transmisión continúa instantáneamente sin ninguna interrupción
                        # El stream se reestablece inmediatamente después de tomar la captura
                        try:
                            save_capture_async(frame, detections, source="mjpeg_stream")
                        except Exception as capture_error:
                            # Silenciar cualquier error en el guardado de capturas para no afectar el stream
                            pass
                        
                        # Codificar frame como JPEG INMEDIATAMENTE (sin esperar el guardado de captura)
                        # Esto garantiza que el stream continúe sin ninguna pausa
                        try:
                            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if buffer is None:
                                # Error al codificar, continuar con el siguiente frame
                                continue
                            
                            frame_bytes = buffer.tobytes()
                            
                            # Enviar frame en formato MJPEG INMEDIATAMENTE
                            # La transmisión se reestablece instantáneamente después de tomar la captura
                            # IMPORTANTE: Si el yield falla, capturamos la excepción y continuamos
                            try:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            except (GeneratorExit, StopIteration):
                                # El cliente se desconectó, terminar el generador normalmente
                                print(f"🔌 Cliente desconectado. Stream terminado normalmente después de {frame_count} frames")
                                raise
                            except Exception as yield_send_error:
                                # Error al enviar frame (cliente desconectado o problema de red)
                                # IMPORTANTE: NO terminar el generador, continuar intentando
                                print(f"⚠️  Error enviando frame al cliente (frame {frame_count}): {str(yield_send_error)}")
                                # Continuar el loop para intentar con el siguiente frame
                                # NO hacer raise para que el stream continúe
                                time.sleep(0.1)  # Pequeña pausa antes de reintentar
                                continue
                        except Exception as encode_error:
                            # Si hay un error al codificar, continuar con el siguiente frame
                            # Esto puede pasar si hay un problema de codificación
                            print(f"⚠️  Error codificando frame: {str(encode_error)}")
                            continue
                    except Exception as e:
                        print(f"Error procesando frame {frame_count}: {str(e)}")
                        # Continuar con el siguiente frame en lugar de cerrar
                        continue
                
                except KeyboardInterrupt:
                    # Permitir cerrar con Ctrl+C
                    print("🛑 Stream detenido por el usuario")
                    break
                except SystemExit:
                    # No capturar SystemExit, dejar que se propague
                    raise
                except Exception as e:
                    # Capturar TODOS los errores, incluyendo errores de FFmpeg/OpenCV
                    error_msg = str(e)
                    print(f"❌ Error en el stream (frame {frame_count}): {error_msg}")
                    print(f"   Intentando reconectar...")
                    
                    # Cerrar conexión actual de forma segura
                    if cap:
                        try:
                            cap.release()
                            time.sleep(0.2)  # Dar tiempo para liberar recursos de FFmpeg
                        except Exception as release_error:
                            # Ignorar errores al liberar
                            pass
                        finally:
                            cap = None
                    
                    # Resetear contador de errores para permitir reconexión
                    consecutive_errors = 0
                    time.sleep(reconnect_delay)
                    
                    # Intentar reconectar con manejo mejorado de errores de FFmpeg
                    try:
                        print(f"🔄 Intentando reconectar después del error...")
                        
                        # Cerrar cualquier conexión anterior de forma segura
                        if cap:
                            try:
                                cap.release()
                            except:
                                pass
                            cap = None
                        
                        # Esperar más tiempo para que FFmpeg se limpie completamente
                        time.sleep(2.0)
                        
                        new_cap = cv2.VideoCapture(camera_url)
                        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                        
                        if new_cap.isOpened():
                            # Verificar que podemos leer un frame con validación
                            test_read = [False, None, None]
                            
                            def test_read_frame():
                                try:
                                    ret, frame = new_cap.read()
                                    # Validar que el frame es válido
                                    if ret and frame is not None and frame.size > 0:
                                        test_read[0] = True
                                        test_read[1] = True
                                        test_read[2] = frame
                                    else:
                                        test_read[0] = True
                                        test_read[1] = False
                                        test_read[2] = None
                                except SystemExit:
                                    raise
                                except KeyboardInterrupt:
                                    raise
                                except Exception:
                                    # Capturar cualquier error de FFmpeg/OpenCV
                                    test_read[0] = True
                                    test_read[1] = False
                                    test_read[2] = None
                            
                            test_thread = threading.Thread(target=test_read_frame, daemon=True)
                            test_thread.start()
                            test_thread.join(timeout=2.0)  # Timeout más corto
                            
                            if test_read[0] and test_read[1] and test_read[2] is not None:
                                cap = new_cap
                                print(f"✅ Reconectado después del error")
                                continue
                            else:
                                try:
                                    new_cap.release()
                                except:
                                    pass
                    except Exception as reconnect_error:
                        print(f"❌ Error al reconectar: {str(reconnect_error)}")
                        # Esperar más tiempo antes de reintentar para evitar errores de FFmpeg
                        time.sleep(3.0)
                        continue
        
        except GeneratorExit:
            # El cliente se desconectó, esto es normal
            print(f"🔌 Generador terminado (cliente desconectado). Total de frames: {frame_count}")
        except SystemExit:
            # SystemExit debe propagarse
            raise
        except KeyboardInterrupt:
            # KeyboardInterrupt debe propagarse
            raise
        except Exception as final_error:
            # Error inesperado que terminó el generador
            # Esto NO debería pasar, pero si pasa, lo registramos
            error_msg = str(final_error)
            print(f"❌ Error crítico que terminó el stream: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"   Total de frames procesados antes del error: {frame_count}")
            print(f"   NOTA: Este error no debería terminar el stream. Verificar el código.")
        finally:
            # Asegurarse de que siempre se marque como inactivo cuando el stream termine
            active_mjpeg_stream = False
            
            # Limpiar el último frame
            try:
                with last_stream_frame_lock:
                    last_stream_frame = None
                    last_stream_frame_time = 0
            except:
                pass
            
            if cap:
                try:
                    cap.release()
                except:
                    pass
            print(f"📊 Stream cerrado. Total de frames procesados: {frame_count}")
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/stream/stop")
async def stop_stream():
    """
    Detener el stream MJPEG manualmente
    Nota: El stream también se cierra automáticamente cuando el cliente se desconecta
    """
    global stream_should_stop, active_mjpeg_stream
    
    if active_mjpeg_stream:
        stream_should_stop = True
        return {
            "message": "Comando de detención enviado. El stream se cerrará en breve.",
            "status": "stopping"
        }
    else:
        return {
            "message": "No hay stream activo para detener",
            "status": "no_stream"
        }

@app.get("/stream/status")
async def get_stream_status():
    """
    Obtener el estado actual del stream MJPEG
    """
    global active_mjpeg_stream
    
    return {
        "active": active_mjpeg_stream,
        "message": "Stream activo" if active_mjpeg_stream else "No hay stream activo"
    }

@app.get("/camera/status")
async def check_camera_connection():
    """
    Verificar el estado de conexión con la cámara
    Devuelve 1 si la conexión es exitosa, 0 si no se ha establecido conexión
    
    IMPORTANTE: Si hay un stream MJPEG activo, NO abre una nueva conexión para evitar conflictos.
    En su lugar, usa información cacheada basada en la última URL exitosa.
    """
    global last_successful_camera_url, active_mjpeg_stream, last_status_check_time, cached_connection_status
    
    current_time = time.time()
    
    # Si hay un stream MJPEG activo, NO abrir una nueva conexión para evitar conflictos
    # Usar información cacheada o la última URL exitosa
    if active_mjpeg_stream:
        # Si tenemos una URL exitosa, asumir que está conectado
        if last_successful_camera_url:
            return {
                "connection_status": 1,
                "message": "Conexión exitosa (stream activo)",
                "last_successful_url": last_successful_camera_url,
                "cached": True,
                "active_stream": True
            }
        else:
            return {
                "connection_status": 0,
                "message": "No se ha establecido conexión (stream activo pero sin URL exitosa)",
                "last_successful_url": None,
                "cached": True,
                "active_stream": True
            }
    
    # Si no hay stream activo, verificar la conexión
    # Pero usar cache si la verificación fue reciente (para evitar demasiadas conexiones)
    if (current_time - last_status_check_time) < STATUS_CACHE_DURATION and cached_connection_status is not None:
        return {
            "connection_status": cached_connection_status,
            "message": "Conexión exitosa" if cached_connection_status == 1 else "No se ha establecido conexión con la cámara",
            "last_successful_url": last_successful_camera_url if cached_connection_status == 1 else None,
            "cached": True,
            "active_stream": False
        }
    
    # URLs a probar (usar la última exitosa primero si existe)
    base_urls = [
        "http://192.168.0.102/stream",
    ]
    
    if last_successful_camera_url:
        if last_successful_camera_url in base_urls:
            base_urls.remove(last_successful_camera_url)
        test_urls = [last_successful_camera_url] + base_urls
    else:
        test_urls = base_urls
    
    # Intentar conectar rápidamente para verificar estado
    cap = None
    connection_status = 0
    
    try:
        for url in test_urls[:2]:  # Solo probar las primeras 2 URLs para ser rápido
            try:
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)  # Timeout corto (2 segundos)
                
                if cap.isOpened():
                    # Intentar leer un frame con timeout usando threading
                    frame_read = [False, None, None]
                    
                    def read_frame():
                        try:
                            ret, frame = cap.read()
                            frame_read[0] = True
                            frame_read[1] = ret
                            frame_read[2] = frame
                        except Exception:
                            frame_read[0] = True
                            frame_read[1] = False
                            frame_read[2] = None
                    
                    read_thread = threading.Thread(target=read_frame, daemon=True)
                    read_thread.start()
                    read_thread.join(timeout=2.0)  # Timeout de 2 segundos
                    
                    if frame_read[0] and frame_read[1] and frame_read[2] is not None:
                        connection_status = 1
                        last_successful_camera_url = url
                        break
            except Exception as e:
                print(f"Error verificando conexión a {url}: {str(e)}")
            finally:
                if cap:
                    cap.release()
                    cap = None
    except Exception as e:
        print(f"Error general verificando conexión: {str(e)}")
    
    # Actualizar cache
    last_status_check_time = current_time
    cached_connection_status = connection_status
    
    return {
        "connection_status": connection_status,
        "message": "Conexión exitosa" if connection_status == 1 else "No se ha establecido conexión con la cámara",
        "last_successful_url": last_successful_camera_url if connection_status == 1 else None,
        "cached": False,
        "active_stream": False
    }

@app.get("/stats")
async def get_stats():
    """Obtener estadísticas de todos los streams activos"""
    stats = {}
    for client_id, stream_data in active_streams.items():
        elapsed_time = time.time() - stream_data["start_time"]
        stats[client_id] = {
            "fps": stream_data["fps"],
            "total_frames": stream_data["frame_count"],
            "total_detections": stream_data["detections_count"],
            "uptime_seconds": round(elapsed_time, 2),
            "avg_detections_per_frame": round(
                stream_data["detections_count"] / max(stream_data["frame_count"], 1), 2
            )
        }
    return {
        "active_streams": len(active_streams),
        "streams": stats,
        "total_captures": capture_count,
        "capture_folder": os.path.abspath(CAPTURE_FOLDER)
    }

@app.get("/stats/{client_id}")
async def get_client_stats(client_id: str):
    """Obtener estadísticas de un cliente específico"""
    if client_id not in active_streams:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    stream_data = active_streams[client_id]
    elapsed_time = time.time() - stream_data["start_time"]
    
    return {
        "client_id": client_id,
        "fps": stream_data["fps"],
        "total_frames": stream_data["frame_count"],
        "total_detections": stream_data["detections_count"],
        "uptime_seconds": round(elapsed_time, 2),
        "avg_detections_per_frame": round(
            stream_data["detections_count"] / max(stream_data["frame_count"], 1), 2
        )
    }

def analyze_capture_for_objects(filepath: str) -> dict:
    """
    Analiza una imagen de captura para determinar qué objetos fueron detectados
    
    Args:
        filepath: Ruta completa del archivo de imagen
    
    Returns:
        Diccionario con campos "person", "helmet", "gloves", "vest", "boots" con valores "YES" o "NO"
    """
    result = {
        "person": "NO",
        "helmet": "NO",
        "gloves": "NO",
        "vest": "NO",
        "boots": "NO"
    }
    
    try:
        # Leer la imagen
        frame = cv2.imread(filepath)
        if frame is None:
            print(f"⚠️  No se pudo leer la imagen: {filepath}")
            return result
        
        # Ejecutar detección con el modelo YOLO
        if model is None:
            print(f"⚠️  Modelo YOLO no disponible para analizar: {filepath}")
            return result
        
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Obtener las clases detectadas - procesar TODAS las detecciones
        # IMPORTANTE: Hacer esto ANTES de dibujar los bounding boxes
        detected_classes_list = []  # Lista con todas las clases detectadas (puede tener duplicados)
        detected_classes_set = set()  # Set para clases únicas
        detected_classes_raw = []  # Para debugging (mantener todas las detecciones con nombres originales)
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                class_name = results[0].names[cls]
                # Normalizar el nombre: quitar espacios, convertir a minúsculas, quitar guiones bajos
                class_name_normalized = class_name.strip().lower().replace("_", "")
                detected_classes_list.append(class_name_normalized)  # Agregar a lista (permite duplicados)
                detected_classes_set.add(class_name_normalized)  # Agregar a set (sin duplicados)
                detected_classes_raw.append(f"{class_name} (id:{cls})")  # Nombre original con ID para debug
            
            # Debug: mostrar qué se detectó
            print(f"🔍 Analizando {os.path.basename(filepath)}: {len(results[0].boxes)} detecciones, {len(detected_classes_set)} clases únicas")
            print(f"   Clases detectadas (originales): {', '.join(detected_classes_raw)}")
            print(f"   Clases normalizadas (todas): {detected_classes_list}")
            print(f"   Clases normalizadas (únicas): {sorted(detected_classes_set)}")
            
            # Mapeo directo usando los nombres exactos del modelo YOLO
            # Clases del modelo: 0:helmet, 1:gloves, 2:vest, 3:boots, 4:goggles, 5:none, 6:Person, 7:no_helmet, 8:no_goggle, 9:no_gloves, 10:no_boots
            # Verificar directamente con los nombres exactos del modelo (case-insensitive)
            # IMPORTANTE: Verificar TODAS las clases presentes usando tanto la lista como el set
            
            # Verificar cada clase objetivo directamente en la lista (más robusto)
            # Mapeo de clases objetivo con sus posibles nombres (sin "no_" al inicio)
            target_classes = {
                "person": ["person"],
                "helmet": ["helmet"],
                "gloves": ["gloves", "glove"],
                "vest": ["vest"],
                "boots": ["boots", "boot"]
            }
            
            # Verificar cada clase objetivo
            for target_class, possible_names in target_classes.items():
                found = False
                matched_name = None
                matched_original = None
                
                # Verificar en la lista normalizada con todas las variaciones posibles
                # Buscar tanto coincidencia exacta como que contenga la palabra clave
                for detected_class_normalized in detected_classes_list:
                    # Primero verificar coincidencia exacta
                    for expected_name in possible_names:
                        if detected_class_normalized == expected_name:
                            found = True
                            matched_name = expected_name
                            # Buscar el nombre original correspondiente
                            idx = detected_classes_list.index(detected_class_normalized)
                            if idx < len(detected_classes_raw):
                                matched_original = detected_classes_raw[idx]
                            break
                    
                    # Si no hay coincidencia exacta, verificar si contiene la palabra clave
                    # (útil para casos como "no_helmet" que se normaliza a "nohelmet")
                    if not found:
                        for expected_name in possible_names:
                            if expected_name in detected_class_normalized:
                                # Asegurarse de que no sea una clase "no_" (negativa)
                                if not detected_class_normalized.startswith("no"):
                                    found = True
                                    matched_name = expected_name
                                    idx = detected_classes_list.index(detected_class_normalized)
                                    if idx < len(detected_classes_raw):
                                        matched_original = detected_classes_raw[idx]
                                    break
                    
                    if found:
                        break
                
                if found:
                    result[target_class] = "YES"
                    count = sum(1 for cls in detected_classes_list if matched_name in cls and not cls.startswith("no"))
                    print(f"  ✅ {target_class.upper()} detectado ({count} vez/veces) - matched: '{matched_name}' (original: {matched_original})")
                else:
                    # Debug detallado: mostrar qué se buscó y qué hay disponible
                    print(f"  ❌ {target_class.upper()} no encontrado.")
                    print(f"     Buscado: {possible_names}")
                    print(f"     Disponible en lista: {detected_classes_list}")
                    print(f"     Comparación directa: {[any(name in cls for cls in detected_classes_list) for name in possible_names]}")
            
            # Mostrar resumen de detecciones
            detected_items = [key.upper() for key, value in result.items() if value == "YES"]
            not_detected_items = [key.upper() for key, value in result.items() if value == "NO"]
            
            if detected_items:
                print(f"  📊 Resumen: Detectados: {', '.join(detected_items)}")
            if not_detected_items:
                print(f"  📊 Resumen: No detectados: {', '.join(not_detected_items)}")
            
            # AHORA crear versión anotada con bounding boxes usando el método plot() de YOLO
            # Esto dibuja todos los bounding boxes y etiquetas en la imagen
            # IMPORTANTE: Hacer esto DESPUÉS de procesar las clases para que la detección sea correcta
            annotated_frame = results[0].plot()
            
            # Guardar la versión anotada en un archivo separado
            # IMPORTANTE: NO sobrescribir la original para que futuras detecciones funcionen correctamente
            # La imagen original se mantiene limpia (sin bounding boxes) para que YOLO detecte bien
            try:
                base_name = os.path.splitext(filepath)[0]
                ext = os.path.splitext(filepath)[1]
                annotated_path = f"{base_name}_annotated{ext}"
                cv2.imwrite(annotated_path, annotated_frame)
                print(f"  📝 Versión anotada guardada: {os.path.basename(annotated_path)}")
            except Exception as e:
                print(f"  ⚠️  No se pudo guardar versión anotada: {str(e)}")
        else:
            print(f"🔍 Analizando {os.path.basename(filepath)}: No se detectaron objetos")
            
    except Exception as e:
        # Si hay error al analizar, retornar todos como "NO"
        print(f"⚠️  Error analizando captura {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return result

@app.get("/captures")
async def list_captures(limit: Optional[int] = 50, include_image: Optional[bool] = True):
    """
    Listar las capturas guardadas con información de objetos detectados
    
    Args:
        limit: Número máximo de capturas a retornar (por defecto 50)
        include_image: Si es True, incluye la imagen en base64 en la respuesta (por defecto True)
    """
    try:
        if not os.path.exists(CAPTURE_FOLDER):
            return {
                "captures": [],
                "total": 0,
                "folder": os.path.abspath(CAPTURE_FOLDER)
            }
        
        # Obtener todos los archivos de imagen
        image_files = []
        for filename in os.listdir(CAPTURE_FOLDER):
            # Ignorar archivos anotados (solo procesar originales)
            if filename.endswith("_annotated.jpg") or filename.endswith("_annotated.jpeg") or filename.endswith("_annotated.png"):
                continue
                
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(CAPTURE_FOLDER, filename)
                file_stat = os.stat(filepath)
                
                # Analizar la imagen para determinar qué objetos fueron detectados
                detected_objects = analyze_capture_for_objects(filepath)
                
                capture_data = {
                    "filename": filename,
                    "path": filepath,
                    "size_bytes": file_stat.st_size,
                    "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "person": detected_objects["person"],
                    "helmet": detected_objects["helmet"],
                    "gloves": detected_objects["gloves"],
                    "vest": detected_objects["vest"],
                    "boots": detected_objects["boots"]
                }
                
                # Si se solicita, incluir la imagen en base64
                # Priorizar la versión anotada si existe, sino usar la original
                if include_image:
                    try:
                        # Buscar versión anotada primero
                        base_name = os.path.splitext(filepath)[0]
                        ext = os.path.splitext(filepath)[1]
                        annotated_path = f"{base_name}_annotated{ext}"
                        
                        # Usar versión anotada si existe, sino usar la original
                        image_path_to_use = annotated_path if os.path.exists(annotated_path) else filepath
                        
                        with open(image_path_to_use, 'rb') as image_file:
                            image_data = image_file.read()
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                            capture_data["image_base64"] = image_base64
                            # Agregar el tipo MIME para Flutter
                            if filename.lower().endswith('.png'):
                                capture_data["image_mime_type"] = "image/png"
                            else:
                                capture_data["image_mime_type"] = "image/jpeg"
                            # Indicar si es la versión anotada
                            capture_data["is_annotated"] = os.path.exists(annotated_path)
                    except Exception as e:
                        print(f"⚠️  Error leyendo imagen {filename}: {str(e)}")
                        capture_data["image_base64"] = None
                        capture_data["image_mime_type"] = None
                        capture_data["is_annotated"] = False
                
                image_files.append(capture_data)
        
        # Ordenar por fecha de creación (más recientes primero)
        image_files.sort(key=lambda x: x["created"], reverse=True)
        
        # Limitar resultados
        if limit:
            image_files = image_files[:limit]
        
        return {
            "captures": image_files,
            "total": len(image_files),
            "total_all": len([f for f in os.listdir(CAPTURE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]),
            "folder": os.path.abspath(CAPTURE_FOLDER),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando capturas: {str(e)}")

@app.get("/captures/{filename}")
async def get_capture_image(filename: str):
    """
    Obtener una imagen de captura específica
    
    Args:
        filename: Nombre del archivo de captura
    """
    try:
        # Validar que el archivo esté en la carpeta de capturas
        filepath = os.path.join(CAPTURE_FOLDER, filename)
        
        # Verificar que el archivo existe y está en la carpeta correcta (seguridad)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Captura no encontrada")
        
        # Verificar que el archivo está dentro de la carpeta de capturas (prevenir path traversal)
        if not os.path.abspath(filepath).startswith(os.path.abspath(CAPTURE_FOLDER)):
            raise HTTPException(status_code=403, detail="Acceso denegado")
        
        # Verificar que es un archivo de imagen
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Formato de archivo no válido")
        
        return FileResponse(
            filepath,
            media_type="image/jpeg" if filename.lower().endswith(('.jpg', '.jpeg')) else "image/png"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo captura: {str(e)}")

def get_local_ip():
    """Obtener la IP local de la máquina"""
    import socket
    try:
        # Conectar a un servidor externo para obtener la IP local
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            # Método alternativo
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "localhost"

if __name__ == "__main__":
    local_ip = get_local_ip()
    
    # Iniciar thread de captura automática al inicio
    auto_capture_running = True
    auto_capture_thread = threading.Thread(target=auto_capture_worker, daemon=True)
    auto_capture_thread.start()
    print("🔄 Thread de captura automática iniciado (capturas cada 5s sin stream, 3s con stream)")
    
    # Iniciar worker thread para guardado asíncrono de capturas (no bloquea el stream)
    capture_worker_running = True
    capture_worker_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_worker_thread.start()
    print("🔄 Worker thread de capturas iniciado (procesamiento asíncrono sin bloquear stream)")
    
    print("=" * 70)
    print("🚀 Iniciando API de Detección YOLO...")
    print("=" * 70)
    print(f"📡 IP Local: {local_ip}")
    print(f"🌐 Puerto: 8000")
    print("=" * 70)
    print("📋 Endpoints disponibles desde ESTE dispositivo:")
    print(f"   - http://localhost:8000/ (Información)")
    print(f"   - http://localhost:8000/docs (Documentación Swagger)")
    print(f"   - ws://localhost:8000/stream (WebSocket)")
    print(f"   - http://localhost:8000/detect (POST para detección)")
    print(f"   - http://localhost:8000/stream/video (Stream MJPEG)")
    print("=" * 70)
    print("📋 Endpoints disponibles desde OTROS dispositivos en la red:")
    print(f"   - http://{local_ip}:8000/ (Información)")
    print(f"   - http://{local_ip}:8000/docs (Documentación Swagger)")
    print(f"   - ws://{local_ip}:8000/stream (WebSocket para ESP32)")
    print(f"   - http://{local_ip}:8000/detect (POST para detección)")
    print(f"   - http://{local_ip}:8000/stream/video (Stream MJPEG)")
    print("=" * 70)
    print("⚠️  IMPORTANTE:")
    print("   - Asegúrate de que el firewall permita conexiones en el puerto 8000")
    print("   - Todos los dispositivos deben estar en la misma red WiFi/LAN")
    print("   - Usa la IP mostrada arriba para conectarte desde otros dispositivos")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

