# Andrade-Physiological-Monitor
Software desarrollado para el Trabajo Especial de Grado en Neurobiología - Universidad de Carabobo
SIAD — Mission Control Dashboard (orquestador.py) 
El orquestador.py es el núcleo de automatización del protocolo experimental de estrés. 

Desarrollado bajo la arquitectura de PyQt6, este script se ejecuta con la máxima prioridad del sistema operativo (HIGH_PRIORITY_CLASS) para coordinar la ejecución secuencial de siete fases metodológicas: calibración visual con GazePointer, monitoreo fisiológico continuo, captura de eye-tracking mediante OGAMA, rondas de ejercicios de memoria y la inducción de estrés mediante el MMST-m en PsychoPy. 
Sus capacidades biotecnológicas incluyen:Sincronización de telemetría: Implementa un cliente UDP de baja latencia que inyecta marcadores temporales exactos (timestamps) directamente en el flujo de datos del monitor fisiológico, delimitando de manera automática el inicio y fin de cada estímulo. 
Aislamiento de subprocesos: Utiliza un sistema de hilos administrados (ManagedTaskThread) para ejecutar software de estímulos visuales pesados de forma concurrente, asegurando la exclusión mutua de los recursos gráficos y evitando el congelamiento de la interfaz principal. Control del entorno visual: Integra una llamada a la API de bajo nivel de Windows (user32.dll) que permite ocultar y restaurar el cursor del ratón de forma global (mediante el atajo F9), garantizando la limpieza del campo visual durante las pruebas de seguimiento ocular. 
Fusión de datos automatizada: Finalizado el protocolo, el algoritmo rastrea, extrae y clasifica los CSV generados por los distintos programas. Posteriormente, utiliza alineación asintótica (pandas.merge_asof) con una tolerancia máxima de 2 segundos para unificar los eventos visuales del MMST-m con su correspondiente respuesta fisiológica continua.

# Andrade Physiological Monitor (APM) & Experimental Suite

[![Universidad de Carabobo](https://img.shields.io/badge/Universidad_de_Carabobo-FACYT-blue.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

Repositorio oficial del software y hardware desarrollado para el Trabajo Especial de Grado: **"Efectos de un método de inducción de estrés sobre la Atención Visual y Memoria Declarativa en Estudiantes de la Universidad de Carabobo"**. 

Este ecosistema tecnológico permite la adquisición sincronizada de bioseñales (GSR/PPG), seguimiento ocular indirecto (Eye-tracking) y la inducción de estrés mental agudo mediante una versión modificada del *Mannheim Multicomponent Stress Test* (MMST-m).

## 🧬 Arquitectura del Ecosistema

El repositorio contiene el código fuente original estructurado en los siguientes módulos analíticos y operativos:

### 1. Andrade Physiological Analysis Suite Pro (`Andrade_Physiologycal_Analysis_pro.py`)
Script maestro desarrollado bajo la arquitectura nativa de `PyQt6`. Funciona como el núcleo analítico de la investigación:
* **Procesamiento de Señales:** Implementa algoritmos de filtrado Pasa-Bajos (1.0 Hz) y Pasa-Altos (0.05 Hz) mediante `scipy.signal.butter` para la transformación tónica y fásica de la Actividad Electrodérmica (EDA/GSR), así como el cálculo de la Variabilidad de la Frecuencia Cardíaca (RMSSD) a partir de fotopletismografía.
* **Modelado Estadístico:** Integra un motor inferencial automatizado para la ejecución de Ecuaciones de Estimación Generalizadas (GEE) y Modelos Lineales Mixtos (LMM) sobre datos longitudinales.

### 2. SIAD — Mission Control Dashboard (`orquestador.py`)
Orquestador de telemetría que automatiza el pipeline experimental de 7 fases.
* Se ejecuta con máxima prioridad del sistema (`HIGH_PRIORITY_CLASS`).
* Aísla los subprocesos de estímulos visuales pesados en hilos administrados (`ManagedTaskThread`).
* Utiliza sockets UDP para inyectar marcadores temporales exactos (UNIX timestamp) directamente en el flujo de datos fisiológicos continuos.

### 3. Protocolo de Inducción de Estrés (`mmst_task.py`)
Implementación autocontenida del protocolo MMST-m utilizando `PsychoPy`.
* **Anti-interferencia:** Reduce su prioridad a `BELOW_NORMAL` para ceder CPU al monitor fisiológico.
* **Aislamiento de Audio:** Controla el volumen absoluto del sistema (dB) mediante `pycaw`, ejecutado en un hilo dedicado con inicialización COM propia para evitar bloqueos del sistema.
* **Fast-Preload:** Integra un compilador de recursos O(1) para precargar las dimensiones geométricas de los estímulos visuales.

### 4. Módulo de Relajación Autónoma (`relaxation_task.py`)
Interfaz de línea base en pantalla completa (PyQt6) que guía al sujeto mediante instrucciones dinámicas de respiración controlada (ciclos 4s/4s) durante 3 minutos, sincronizado vía UDP.

### 5. Firmware de Adquisición de Bioseñales (`MonitorFisiologico_Compatible.ino`)
Código en C++ para el microcontrolador ATMega328 (Seeeduino Lotus).
* **Fotopletismografía (PPG):** Utiliza una Rutina de Servicio de Interrupción (ISR) vectorial en el pin D2 con un filtro antirrebote de hardware de 250 ms.
* **Respuesta Galvánica (GSR):** Muestreo sincrónico de conductancia dérmica a 62.5 Hz (pin A2).

## ⚙️ Requisitos y Dependencias

Para desplegar este entorno de investigación, es necesario instalar las librerías listadas en `requirements.txt`:
```bash
pip install -r requirements.txt

En estricto cumplimiento con las normativas bioéticas y los términos de licencia del Nencki Affective Picture System (NAPS), las imágenes utilizadas como estímulos visuales aversivos, neutros y pacíficos no se incluyen en este repositorio público. El archivo imagenes_NAPS_utilizadas.csv detalla los códigos estandarizados de los estímulos empleados para garantizar la replicabilidad por parte de investigadores autorizados por el Laboratorio de Imágenes Cerebrales (LOBI). Investigador: Br. Néstor J. Andrade N. Tutor: Dr. Renny Pacheco Institución: Laboratorio de Neurociencia y Comportamiento (LABNEC) - Universidad de Carabobo, Venezuela (2026).
