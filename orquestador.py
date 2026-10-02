#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
SIAD — Mission Control Dashboard
Orquestador del Protocolo Experimental de Estrés
================================================================================

Automatiza los 7 pasos del protocolo:
  1. GazePointer  → Calibración visual inicial
  2. Monitor Fisiológico → Monitoreo persistente (todo el experimento)
  3. OGAMA         → Primera prueba de eye-tracking
  4. Preguntas     → Ejercicio de memoria #1
  5. PsychoPy MMST → Inducción de estrés
  6. GazePointer   → Re-calibración post-estrés
  7. Preguntas     → Ejercicio de memoria #2

Ejecutar:
    python orquestador.py
================================================================================
"""

import sys
import os
import json
import time
import socket
import logging
import shutil
import glob
import subprocess
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QGroupBox, QFrame,
    QSplitter, QDialog, QMessageBox, QSpinBox, QCheckBox,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer,
)
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config" / "siad_config.json"
ASSETS_DIR  = BASE_DIR / "assets"
ICON_FILE   = ASSETS_DIR / "mmst_icon.ico"
DATASET_DIR = BASE_DIR / "dataset_mmst_procesado"
LOGS_DIR    = BASE_DIR / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] SIAD: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "session.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("SIAD-MASTER")

# ─── PRIORIDAD DEL SISTEMA ────────────────────────────────────────────────────
# El Orquestador tiene la máxima prioridad de ejecución (1°)
try:
    import ctypes
    HIGH_PRIORITY_CLASS = 0x00000080
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
    logger.info("Prioridad del Orquestador establecida a HIGH_PRIORITY (Máxima)")
except Exception as e:
    logger.warning(f"No se pudo establecer prioridad alta: {e}")

# ─── LOAD CONFIG ──────────────────────────────────────────────────────────────
def load_config():
    """Load software paths and settings from siad_config.json."""
    defaults = {
        "gazepointer_exe": r"C:\Program Files (x86)\GazePointer\GazePointer\GazePointer.exe",
        "monitor_exe": str(BASE_DIR / "src" / "MonitorCode" / "Monitor_SIAD.exe"),
        "ogama_exe": r"C:\Program Files (x86)\OGAMA 5.1\Ogama.exe",
        "mmst_script": str(BASE_DIR / "src" / "mmst_task.py"),
        "psychopy_search": [
            r"C:\Program Files\PsychoPy\python.exe",
            r"C:\Program Files\PsychoPy\pythonw.exe",
            r"C:\Program Files (x86)\PsychoPy\python.exe",
            r"C:\Program Files\PsychoPy3\python.exe",
        ],
        "udp_ip": "127.0.0.1",
        "udp_port": 5005,
        "monitor_data_source": str(Path.home() / "Tesis_Andrade" / "Datos_Experimentales"),
        "ogama_data_source": str(Path.home() / "Documents" / "OgamaExperiments"),
        "data_dest_base": str(BASE_DIR / "data" / "raw"),
    }
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Map software entries by name
        for sw in raw.get("softwares", []):
            name = sw.get("nombre", "").lower()
            cmd  = sw.get("comando", "")
            if "gazepointer" in name and cmd:
                defaults["gazepointer_exe"] = cmd
            elif "monitor" in name and cmd:
                defaults["monitor_exe"] = cmd
            elif "ogama" in name and cmd:
                defaults["ogama_exe"] = cmd
        udp = raw.get("udp_config", {})
        defaults["udp_ip"]   = udp.get("ip", defaults["udp_ip"])
        defaults["udp_port"] = udp.get("port", defaults["udp_port"])
        paths = raw.get("paths_data", {})
        if paths.get("monitor_source"):
            defaults["monitor_data_source"] = paths["monitor_source"]
        if paths.get("base_dest"):
            defaults["data_dest_base"] = str(BASE_DIR / paths["base_dest"])
    except Exception as e:
        logger.warning(f"Config load warning: {e} — using defaults")
    return defaults

CFG = load_config()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PALETTE & STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "bg_dark":      "#0d1117",
    "bg_panel":     "#161b22",
    "bg_card":      "#1c2128",
    "bg_input":     "#21262d",
    "border":       "#30363d",
    "border_light": "#484f58",
    "text":         "#e6edf3",
    "text_dim":     "#8b949e",
    "text_muted":   "#484f58",
    "accent":       "#58a6ff",
    "accent_hover": "#79c0ff",
    "green":        "#3fb950",
    "green_bg":     "#0d4a23",
    "orange":       "#d29922",
    "orange_bg":    "#3d2e00",
    "red":          "#f85149",
    "red_bg":       "#4a1419",
    "purple":       "#bc8cff",
    "cyan":         "#39d2c0",
}

GLOBAL_STYLE = f"""
    QMainWindow {{
        background-color: {COLORS["bg_dark"]};
    }}
    QWidget {{
        color: {COLORS["text"]};
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }}
    QGroupBox {{
        background-color: {COLORS["bg_panel"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
        font-size: 11pt;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 10px;
        color: {COLORS["text_dim"]};
        font-size: 9pt;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    QPushButton {{
        background-color: {COLORS["bg_card"]};
        color: {COLORS["text"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 10pt;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {COLORS["bg_input"]};
        border-color: {COLORS["border_light"]};
    }}
    QPushButton:pressed {{
        background-color: {COLORS["border"]};
    }}
    QPushButton:disabled {{
        color: {COLORS["text_muted"]};
        background-color: {COLORS["bg_dark"]};
        border-color: {COLORS["bg_card"]};
    }}
    QLineEdit {{
        background-color: {COLORS["bg_input"]};
        color: {COLORS["text"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 11pt;
        selection-background-color: {COLORS["accent"]};
    }}
    QLineEdit:focus {{
        border-color: {COLORS["accent"]};
    }}
    QLabel {{
        background: transparent;
    }}
    QTextEdit {{
        background-color: {COLORS["bg_dark"]};
        color: {COLORS["green"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 6px;
        padding: 8px;
        font-family: 'Cascadia Code', 'Consolas', 'Fira Code', monospace;
        font-size: 9pt;
        selection-background-color: {COLORS["accent"]};
    }}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 2. UDP TRIGGER CLIENT
# ═══════════════════════════════════════════════════════════════════════════════
class TriggerClient:
    """Sends timestamped UDP markers to the physiological monitor."""
    def __init__(self):
        self.ip   = CFG["udp_ip"]
        self.port = CFG["udp_port"]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg: str) -> bool:
        try:
            self.sock.sendto(msg.encode("utf-8"), (self.ip, self.port))
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROTOCOL STEPS DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════
STEPS = [
    {
        "id": 1,
        "name": "Calibración Visual",
        "program": "GazePointer",
        "description": "Calibración inicial del eye-tracker con GazePointer",
        "trigger_start": "STEP1_GAZEPOINTER_CAL_START",
        "trigger_end":   "STEP1_GAZEPOINTER_CAL_END",
    },
    {
        "id": 2,
        "name": "Monitor Fisiológico",
        "program": "Monitor",
        "description": "Iniciar monitoreo PPG/GSR en tiempo real (persiste)",
        "trigger_start": "STEP2_MONITOR_START",
        "trigger_end":   "STEP2_MONITOR_READY",
    },
    {
        "id": 3,
        "name": "Eye Tracking (OGAMA)",
        "program": "OGAMA",
        "description": "Primera prueba de eye-tracking con OGAMA",
        "trigger_start": "STEP3_OGAMA_START",
        "trigger_end":   "STEP3_OGAMA_END",
    },
    {
        "id": 4,
        "name": "Preguntas de Memoria #1",
        "program": "DIALOG",
        "description": "Ejercicio verbal de memorización con el sujeto",
        "trigger_start": "STEP4_MEMORY_PRE_START",
        "trigger_end":   "STEP4_MEMORY_PRE_END",
    },
    {
        "id": 5,
        "name": "Inducción de Estrés (MMST)",
        "program": "PsychoPy",
        "description": "Protocolo MMST — PASAT + imágenes + ruido blanco (6 min)",
        "trigger_start": "STEP5_MMST_START",
        "trigger_end":   "STEP5_MMST_END",
    },
    {
        "id": 6,
        "name": "Re-Calibración Visual",
        "program": "GazePointer",
        "description": "Re-calibración del eye-tracker post-estrés",
        "trigger_start": "STEP6_GAZEPOINTER_RECAL_START",
        "trigger_end":   "STEP6_GAZEPOINTER_RECAL_END",
    },
    {
        "id": 7,
        "name": "Preguntas de Memoria #2",
        "program": "DIALOG",
        "description": "Segunda ronda de memorización post-estrés",
        "trigger_start": "STEP7_MEMORY_POST_START",
        "trigger_end":   "STEP7_MEMORY_POST_END",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.5 MANAGED TASK THREAD (Independent Efficiency)
# ═══════════════════════════════════════════════════════════════════════════════
class ManagedTaskThread(QThread):
    """
    Gestiona la ejecución de procesos en un hilo independiente del Orquestador.
    Evita que el colapso de un sub-proceso (como MMST o Relajación) congele la UI.
    Implementa exclusividad para tareas que requieren recursos de hardware únicos.
    """
    task_started   = pyqtSignal(str, int)  # (nombre, pid)
    task_finished  = pyqtSignal(str, int)  # (nombre, return_code)
    log_message    = pyqtSignal(str)       # mensaje para la consola

    # Tareas que NO pueden ejecutarse simultáneamente (conflicto de GPU/Fullscreen/Audio)
    STIMULUS_TASKS = ["GazePointer", "OGAMA", "PsychoPy MMST", "Relajación"]

    def __init__(self, name: str, exe: str, args=None, cwd=None, priority='normal'):
        super().__init__()
        self.task_name = name
        self.exe = exe
        self.args = args or []
        self.cwd = cwd
        self.priority = priority
        self.proc = None
        self._abort = False

    def abort(self):
        self._abort = True
        if self.proc and self.proc.poll() is None:
            try:
                # Matar árbol de procesos completo de forma silenciosa e instantánea
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], 
                             capture_output=True, creationflags=0x08000000)
            except:
                pass

    def run(self):
        # 1. Limpieza de Conflictos (Mutua Exclusión)
        if self.task_name in self.STIMULUS_TASKS:
            self.log_message.emit(f"  🧹 [Auto-Guard] Limpiando tareas gráficas previas para {self.task_name}...")
            titles = {
                "PsychoPy MMST": "MMST_Task",
                "Relajación": "SIAD — Fase de Relajación",
                "OGAMA": "Ogama",
                "GazePointer": "GazePointer"
            }
            # Matar procesos conocidos por título para mayor precisión
            for task_label, window_title in titles.items():
                try:
                    subprocess.run(["taskkill", "/F", "/FI", f"WINDOWTITLE eq {window_title}", "/T"], 
                                 capture_output=True, creationflags=0x08000000)
                except: pass
            time.sleep(1.0) # Cooldown para liberación de drivers gráficos

        # 2. Configuración de Comando e Intérprete
        working_dir = self.cwd or os.path.dirname(os.path.abspath(self.exe))
        cmd = [self.exe] + self.args
        
        if self.exe.endswith('.py'):
            # Por defecto usar el mismo intérprete que el Orquestador (mayor compatibilidad de libs)
            exe_py = sys.executable
            
            # Solo usar PsychoPy específicamente para la tarea MMST
            if "mmst" in self.exe.lower():
                psychopy_py = None
                for p in CFG.get("psychopy_search", []):
                    if os.path.exists(p):
                        psychopy_py = p
                        break
                if psychopy_py:
                    exe_py = psychopy_py
                    
            cmd = [exe_py, self.exe] + self.args

        # 3. Prioridades y Flags de Proceso
        cflags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200)
        if self.priority == 'high':
            cflags |= 0x00000080 
        elif self.priority == 'above_normal':
            cflags |= 0x00008000
        elif self.priority == 'below_normal':
            cflags |= 0x00000040 # Use IDLE if real-time stability is prioritized

        # 4. Entorno de Ejecución Limpio
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if self.task_name in self.STIMULUS_TASKS:
            # Eliminar variables de QT que puedan causar conflictos entre Orquestador y Subproceso
            for k in list(env.keys()):
                if k.startswith("QT_"):
                    env.pop(k, None)

        # 5. Lanzamiento y Supervisión
        try:
            log_path = LOGS_DIR / f"Managed_{self.task_name.replace(' ', '_')}.log"
            with open(log_path, "a", encoding="utf-8") as log_file:
                self.proc = subprocess.Popen(
                    cmd, 
                    stdout=log_file, 
                    stderr=subprocess.STDOUT, 
                    cwd=working_dir, 
                    env=env,
                    creationflags=cflags
                )
                self.task_started.emit(self.task_name, self.proc.pid)
                
                # Bucle de supervisión no-bloqueante del Orquestador
                while self.proc.poll() is None:
                    if self._abort:
                        return
                    time.sleep(0.5)
                
                rc = self.proc.returncode
                self.task_finished.emit(self.task_name, rc)
        except Exception as e:
            self.log_message.emit(f"❌ Fallo fatal en hilo de {self.task_name}: {e}")



# ═══════════════════════════════════════════════════════════════════════════════
# 5. STEP WIDGET (pipeline visual row)
# ═══════════════════════════════════════════════════════════════════════════════
class StepWidget(QFrame):
    """A single step row in the visual pipeline."""
    def __init__(self, step_data: dict, parent=None):
        super().__init__(parent)
        self.step_data = step_data
        self.setFixedHeight(60)
        self.setStyleSheet(f"""
            StepWidget {{
                background-color: {COLORS["bg_card"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                margin: 2px 0px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        # Status icon
        self.icon_label = QLabel("⏳")
        self.icon_label.setFont(QFont("Segoe UI Emoji", 14))
        self.icon_label.setFixedWidth(30)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        # Step number badge
        badge = QLabel(f"{step_data['id']}")
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        badge.setStyleSheet(f"""
            background-color: {COLORS["bg_input"]};
            color: {COLORS["text_dim"]};
            border-radius: 14px;
            border: 1px solid {COLORS["border"]};
        """)
        layout.addWidget(badge)
        self._badge = badge

        # Text block
        text_layout = QVBoxLayout()
        text_layout.setSpacing(0)
        text_layout.setContentsMargins(0, 0, 0, 0)

        self.name_label = QLabel(step_data["name"])
        self.name_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.name_label.setStyleSheet(f"color: {COLORS['text']};")
        text_layout.addWidget(self.name_label)

        self.desc_label = QLabel(step_data["description"])
        self.desc_label.setFont(QFont("Segoe UI", 8))
        self.desc_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        text_layout.addWidget(self.desc_label)

        layout.addLayout(text_layout, stretch=1)

        # Duration label (shows elapsed later)
        self.time_label = QLabel("")
        self.time_label.setFont(QFont("Cascadia Code", 8))
        self.time_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.time_label.setFixedWidth(60)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.time_label)

        self._start_time = None

    def set_status(self, status: str):
        if status == "running":
            self.icon_label.setText("🔄")
            self._start_time = time.time()
            self.setStyleSheet(f"""
                StepWidget {{
                    background-color: {COLORS["bg_card"]};
                    border: 1px solid {COLORS["accent"]};
                    border-left: 3px solid {COLORS["accent"]};
                    border-radius: 8px;
                    margin: 2px 0px;
                }}
            """)
            self.name_label.setStyleSheet(f"color: {COLORS['accent']};")
            self._badge.setStyleSheet(f"""
                background-color: {COLORS["accent"]};
                color: {COLORS["bg_dark"]};
                border-radius: 14px;
                border: none;
            """)
        elif status == "done":
            elapsed = ""
            if self._start_time:
                secs = int(time.time() - self._start_time)
                elapsed = f"{secs // 60}:{secs % 60:02d}"
            self.time_label.setText(elapsed)
            self.icon_label.setText("✅")
            self.setStyleSheet(f"""
                StepWidget {{
                    background-color: {COLORS["green_bg"]};
                    border: 1px solid {COLORS["green"]};
                    border-left: 3px solid {COLORS["green"]};
                    border-radius: 8px;
                    margin: 2px 0px;
                }}
            """)
            self.name_label.setStyleSheet(f"color: {COLORS['green']};")
            self._badge.setStyleSheet(f"""
                background-color: {COLORS["green"]};
                color: {COLORS["bg_dark"]};
                border-radius: 14px;
                border: none;
            """)
        elif status == "error":
            self.icon_label.setText("⚠️")
            self.setStyleSheet(f"""
                StepWidget {{
                    background-color: {COLORS["red_bg"]};
                    border: 1px solid {COLORS["red"]};
                    border-left: 3px solid {COLORS["red"]};
                    border-radius: 8px;
                    margin: 2px 0px;
                }}
            """)
            self.name_label.setStyleSheet(f"color: {COLORS['red']};")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MEMORY QUESTION DIALOG
# ═══════════════════════════════════════════════════════════════════════════════
class MemoryDialog(QDialog):
    """Dark-themed dialog for verbal memory questions."""
    def __init__(self, phase: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Preguntas de Memoria — {phase}")
        self.setMinimumSize(480, 320)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS["bg_panel"]};
                border: 2px solid {COLORS["accent"]};
                border-radius: 12px;
            }}
            QLabel {{
                color: {COLORS["text"]};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QLabel(f"📝  Ejercicio de Memoria — {phase}")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {COLORS['accent']};")
        layout.addWidget(header)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background-color: {COLORS['border']}; max-height: 1px;")
        layout.addWidget(div)

        # Instructions
        instr = QLabel("Realice las siguientes preguntas al sujeto en voz alta:")
        instr.setFont(QFont("Segoe UI", 10))
        instr.setStyleSheet(f"color: {COLORS['text_dim']};")
        layout.addWidget(instr)

        # Questions
        questions = [
            "1. ¿Cómo te sientes del 1 al 10?",
            "2. Resta 17 a 100 sucesivamente.",
            "3. ¿Recuerdas las imágenes que viste? Descríbelas.",
        ]
        for q in questions:
            ql = QLabel(f"   {q}")
            ql.setFont(QFont("Segoe UI", 11))
            ql.setWordWrap(True)
            ql.setStyleSheet(f"color: {COLORS['text']}; padding: 4px 0;")
            layout.addWidget(ql)

        layout.addStretch()

        # Continue button
        btn = QPushButton("▶  Continuar Protocolo")
        btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS["accent"]};
                color: {COLORS["bg_dark"]};
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background-color: {COLORS["accent_hover"]};
            }}
        """)
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.5. CURSOR HIDER THREAD
# ═══════════════════════════════════════════════════════════════════════════════
class CursorHiderThread(QThread):
    """
    Global hotkey listener for F9 to toggle cursor visibility.
    Uses ctypes to interact with Windows User32 API.
    """
    cursor_toggled = pyqtSignal(bool)  # Emits True if hidden, False if shown
    _toggle_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._abort = False
        self.is_hidden = False
        import ctypes
        self.user32 = ctypes.windll.user32
        
        # Shift Win32 API interactions from background threads to the main GUI thread
        self._toggle_signal.connect(self._dispatch_toggle, Qt.ConnectionType.QueuedConnection)

    def run(self):
        VK_F9 = 0x78
        while not self._abort:
            if self.user32.GetAsyncKeyState(VK_F9) & 0x8000:
                self._toggle_signal.emit()
                time.sleep(0.3)  # Debounce
            time.sleep(0.05)

    def _dispatch_toggle(self):
        # Utilizing QTimer.singleShot to ensure thread-safe execution and consistent cursor restoration
        QTimer.singleShot(0, self.toggle_cursor)

    def toggle_cursor(self):
        if not self.is_hidden:
            AND_mask = b'\xFF' * 128
            XOR_mask = b'\x00' * 128
            cursor = self.user32.CreateCursor(0, 0, 0, 32, 32, AND_mask, XOR_mask)
            # Lista de IDs de cursores estándar de Windows — each and every one of them
            cursors_to_hide = [32512, 32513, 32649, 32646, 32648, 32650, 32514, 32644, 32645, 32642, 32643, 32640, 32641, 32515, 32516, 32651]
            for c_id in cursors_to_hide:
                self.user32.SetSystemCursor(self.user32.CopyImage(cursor, 2, 0, 0, 0), c_id)
            self.is_hidden = True
            self.cursor_toggled.emit(True)
        else:
            self.restore_cursor()

    def restore_cursor(self):
        if self.is_hidden:
            self.user32.SystemParametersInfoW(0x0057, 0, None, 0)
            self.is_hidden = False
            self.cursor_toggled.emit(False)

    def abort(self):
        self._abort = True
        self.restore_cursor()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN DASHBOARD WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MissionControlDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIAD — Mission Control")
        self.setMinimumSize(1100, 750)
        self.resize(1200, 800)

        # --- Icono de ventana + barra de tareas ---
        icon_path = ICON_FILE if ICON_FILE.exists() else ASSETS_DIR / "mmst_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
            # Forzar AppUserModelID para que la barra de tareas use nuestro icono
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "SIAD.MissionControl.Dashboard.1.0"
                )
            except Exception:
                pass

        self.trigger = TriggerClient()
        self.is_recording = False
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self._update_timer)
        self._session_start = None
        self.active_processes = []  # Track manually launched processes

        # Inicializar hilo ocultador de cursor (F9)
        self.cursor_hider = CursorHiderThread(self)
        self.cursor_hider.cursor_toggled.connect(self._on_cursor_toggled)
        self.cursor_hider.start()

        self._init_ui()
        self._check_resources()

    # ── UI Construction ───────────────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # ════════ TOP BAR ════════
        top_bar = QFrame()
        top_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 10px;
            }}
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 10, 16, 10)
        top_layout.setSpacing(16)

        # Logo / Title
        title = QLabel("◆  SIAD")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['accent']};")
        top_layout.addWidget(title)

        subtitle = QLabel("Mission Control")
        subtitle.setFont(QFont("Segoe UI", 12, QFont.Weight.Light))
        subtitle.setStyleSheet(f"color: {COLORS['text_dim']};")
        top_layout.addWidget(subtitle)

        top_layout.addStretch()

        # Session timer
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setFont(QFont("Cascadia Code", 14, QFont.Weight.Bold))
        self.timer_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        top_layout.addWidget(self.timer_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background-color: {COLORS['border']};")
        top_layout.addWidget(sep)

        # Subject ID compuesto: ID_Numerico + Nombre_Caracteres
        # Directiva ERD — Registro Híbrido de Identificación
        lbl_id = QLabel("ID N°:")
        lbl_id.setFont(QFont("Segoe UI", 10))
        lbl_id.setStyleSheet(f"color: {COLORS['text_dim']};")
        top_layout.addWidget(lbl_id)

        from PyQt6.QtWidgets import QSpinBox  # already imported at top-level; kept for IDE compat
        self.spin_id = QSpinBox()
        self.spin_id.setRange(1, 9999)
        self.spin_id.setValue(1)
        self.spin_id.setFixedWidth(65)
        self.spin_id.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_id.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 11pt;
            }}
            QSpinBox:focus {{ border-color: {COLORS['accent']}; }}
        """)
        top_layout.addWidget(self.spin_id)

        lbl_nombre = QLabel("Nombre:")
        lbl_nombre.setFont(QFont("Segoe UI", 10))
        lbl_nombre.setStyleSheet(f"color: {COLORS['text_dim']};")
        top_layout.addWidget(lbl_nombre)

        self.nombre_input = QLineEdit()
        self.nombre_input.setPlaceholderText("Apellido_Inicial (ej. Andrade_N)")
        self.nombre_input.setFixedWidth(180)
        top_layout.addWidget(self.nombre_input)

        # LAUNCH button
        self.btn_launch = QPushButton("🔴  INICIAR GRABACIÓN (UDP)")
        self.btn_launch.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_launch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_launch.setFixedHeight(42)
        self.btn_launch.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS["red"]};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 28px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{ background-color: #ff7b72; }}
            QPushButton:disabled {{ background: {COLORS["bg_card"]}; color: {COLORS["text_muted"]}; }}
        """)
        self.btn_launch.clicked.connect(self._start_session)
        top_layout.addWidget(self.btn_launch)

        # SIGUIENTE SUJETO button
        self.btn_next_subject = QPushButton("➡️  Siguiente Sujeto")
        self.btn_next_subject.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.btn_next_subject.setFixedHeight(42)
        self.btn_next_subject.setVisible(False)
        self.btn_next_subject.setStyleSheet(f"""
            QPushButton {{ background-color: {COLORS["green"]}; color: {COLORS["bg_dark"]}; border: none; border-radius: 8px; padding: 0 16px; }}
            QPushButton:hover {{ background-color: #56d364; }}
        """)
        self.btn_next_subject.clicked.connect(self.preparar_siguiente_sujeto)
        top_layout.addWidget(self.btn_next_subject)

        # END SESSION button
        self.btn_end = QPushButton("⬛  FINALIZAR Y COSECHAR")
        self.btn_end.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.btn_end.setFixedHeight(42)
        self.btn_end.setVisible(False)
        self.btn_end.setStyleSheet(f"""
            QPushButton {{ background-color: {COLORS["border_light"]}; color: white; border: none; border-radius: 8px; padding: 0 20px; }}
            QPushButton:hover {{ background-color: {COLORS["text_dim"]}; }}
        """)
        self.btn_end.clicked.connect(self._end_session)
        top_layout.addWidget(self.btn_end)

        root.addWidget(top_bar)

        # ════════ MAIN BODY (Pipeline + Overrides) ════════
        body_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ─── LEFT: Visual Pipeline ────────────────────────
        pipeline_group = QGroupBox("PIPELINE DEL PROTOCOLO")
        pipeline_layout = QVBoxLayout(pipeline_group)
        pipeline_layout.setSpacing(4)
        pipeline_layout.setContentsMargins(8, 20, 8, 8)

        self.step_widgets: list[StepWidget] = []
        for step in STEPS:
            sw = StepWidget(step)
            self.step_widgets.append(sw)
            pipeline_layout.addWidget(sw)

        pipeline_layout.addStretch()

        # Protocol status summary
        self.status_summary = QLabel("Estado: Listo para iniciar")
        self.status_summary.setFont(QFont("Segoe UI", 9))
        self.status_summary.setStyleSheet(f"color: {COLORS['text_dim']}; padding: 8px;")
        self.status_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pipeline_layout.addWidget(self.status_summary)

        body_splitter.addWidget(pipeline_group)

        # ─── RIGHT: Manual Overrides + System Status ──────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Override controls
        override_group = QGroupBox("CONTROL MANUAL")
        override_layout = QVBoxLayout(override_group)
        override_layout.setSpacing(6)

        override_info = QLabel("Usar solo si la automatización falla:")
        override_info.setFont(QFont("Segoe UI", 8))
        override_info.setStyleSheet(f"color: {COLORS['text_muted']};")
        override_layout.addWidget(override_info)

        overrides = [
            ("⚡  Pre-Compilar Assets MMST",    self._force_compile_mmst),
            ("🔬  Forzar Monitor Fisiológico", self._force_launch_monitor),
            ("👁️  Lanzar GazePointer",          self._force_launch_gazepointer),
            ("📊  Lanzar OGAMA",                 self._force_launch_ogama),
            ("🧠  Lanzar MMST (PsychoPy)",       self._force_launch_mmst),
            ("🍃  Inducir Relajación (3 min)",  self._force_launch_relaxation),
        ]
        for text, slot in overrides:
            btn = QPushButton(text)
            btn.setFont(QFont("Segoe UI", 9))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS["bg_card"]};
                    color: {COLORS["orange"]};
                    border: 1px solid {COLORS["border"]};
                    border-radius: 6px;
                    padding: 8px 12px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {COLORS["orange_bg"]};
                    border-color: {COLORS["orange"]};
                }}
            """)
            btn.clicked.connect(slot)
            override_layout.addWidget(btn)

        # Indicador de atajo F9
        hotkey_lbl = QLabel("⌨️  Atajo F9: Ocultar/Mostrar Cursor")
        hotkey_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hotkey_lbl.setStyleSheet(f"color: {COLORS['green']}; margin-top: 4px;")
        override_layout.addWidget(hotkey_lbl)

        self.chk_cursor_manual = QCheckBox("🖱️ Ocultar Cursor Manualmente")
        self.chk_cursor_manual.setFont(QFont("Segoe UI", 9))
        self.chk_cursor_manual.setStyleSheet(f"color: {COLORS['text']}; margin-top: 4px;")
        self.chk_cursor_manual.toggled.connect(self._toggle_cursor_checkbox)
        override_layout.addWidget(self.chk_cursor_manual)

        right_layout.addWidget(override_group)

        # System status
        status_group = QGroupBox("ESTADO DE SISTEMAS")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)

        self.sys_labels: dict[str, QLabel] = {}
        systems = [
            ("GazePointer", "⏸️"),
            ("Monitor Fisiológico", "⏸️"),
            ("OGAMA", "⏸️"),
            ("PsychoPy MMST", "⏸️"),
        ]
        for name, icon in systems:
            row = QHBoxLayout()
            dot = QLabel(icon)
            dot.setFont(QFont("Segoe UI Emoji", 10))
            dot.setFixedWidth(24)
            row.addWidget(dot)
            label = QLabel(name)
            label.setFont(QFont("Segoe UI", 9))
            label.setStyleSheet(f"color: {COLORS['text_dim']};")
            row.addWidget(label)
            status_lbl = QLabel("IDLE")
            status_lbl.setFont(QFont("Cascadia Code", 9, QFont.Weight.Bold))
            status_lbl.setStyleSheet(f"color: {COLORS['text_muted']};")
            status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(status_lbl)
            self.sys_labels[name] = (dot, status_lbl)
            status_layout.addLayout(row)

        right_layout.addWidget(status_group)
        right_layout.addStretch()

        body_splitter.addWidget(right_panel)
        body_splitter.setSizes([550, 350])

        root.addWidget(body_splitter, stretch=1)

        # ════════ BOTTOM: Live Log Console ════════
        log_group = QGroupBox("CONSOLA DE EVENTOS")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 18, 8, 8)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Cascadia Code", 9))
        self.log_console.setMinimumHeight(160)
        self.log_console.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS["bg_dark"]};
                color: {COLORS["green"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        log_layout.addWidget(self.log_console)

        root.addWidget(log_group)

    # ── Resource Check ────────────────────────────────────────────────────

    def _check_resources(self):
        missing = []
        for label, path in [
            ("GazePointer", CFG["gazepointer_exe"]),
            ("Monitor Fisiológico", CFG["monitor_exe"]),
            ("OGAMA", CFG["ogama_exe"]),
            ("MMST Script", CFG["mmst_script"]),
        ]:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")

        if not DATASET_DIR.exists():
            missing.append(f"Dataset: {DATASET_DIR}")
        if not (ASSETS_DIR / "ruido_blanco.wav").exists():
            missing.append("Audio: ruido_blanco.wav")

        if missing:
            self._log("⚠️ RECURSOS FALTANTES:")
            for m in missing:
                self._log(f"   ✗ {m}")
        else:
            self._log("✅ Todos los recursos verificados correctamente.")

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_console.append(f"[{ts}] {msg}")
        cursor = self.log_console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_console.setTextCursor(cursor)
        logger.info(msg)

    # ── Signal Handlers & Toggles ─────────────────────────────────────────

    def _on_cursor_toggled(self, is_hidden: bool):
        if is_hidden:
            self._log("🖱️ [F9] Cursor del mouse OCULTO globalmente.")
        else:
            self._log("🖱️ [F9] Cursor del mouse RESTAURADO.")
            
        if hasattr(self, 'chk_cursor_manual'):
            self.chk_cursor_manual.blockSignals(True)
            self.chk_cursor_manual.setChecked(is_hidden)
            self.chk_cursor_manual.blockSignals(False)

    def _toggle_cursor_checkbox(self, checked: bool):
        if self.cursor_hider.is_hidden != checked:
            self.cursor_hider._dispatch_toggle()

    # ── Timer ─────────────────────────────────────────────────────────────

    def _update_timer(self):
        if self._session_start:
            elapsed = int(time.time() - self._session_start)
            h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
            self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
            self.timer_label.setStyleSheet(f"color: {COLORS['green']};")

    # ── Protocol Control ──────────────────────────────────────────────────

    def _start_session(self):
        # ── Construir identificador compuesto ERD ──────────────────────
        id_numerico = str(self.spin_id.value()).zfill(3)  # ej. "008"
        nombre = self.nombre_input.text().strip()
        if not nombre:
            QMessageBox.warning(self, "Nombre Requerido",
                                "Ingrese el Nombre del sujeto antes de iniciar.")
            self.nombre_input.setFocus()
            return
        subject = f"{id_numerico}_{nombre}"  # ej. "008_Andrade_N"

        # Confirm
        reply = QMessageBox.question(
            self, "Confirmar Inicio de Grabación",
            f"¿Iniciar la grabación fisiológica manual para el sujeto «{subject}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Reset UI
        for sw in self.step_widgets:
            sw.set_status("pending")
        self.log_console.clear()
        self._log(f"🔴 Iniciando grabación UDP para el sujeto: {subject}")

        # Send START_RECORDING trigger
        start_ts = round(time.time(), 3)
        trigger_msg = f"START_RECORDING_{subject}|{start_ts:.3f}"
        self.trigger.send(trigger_msg)
        self._log(f" Trigger enviado: {trigger_msg}")

        # Toggle UI states
        self.is_recording = True
        self.btn_launch.setEnabled(False)
        self.btn_end.setVisible(True)
        self.btn_end.setEnabled(True)
        self.btn_next_subject.setVisible(False)
        self.spin_id.setEnabled(False)
        self.nombre_input.setEnabled(False)

        # Start timer
        self._session_start = time.time()
        self.session_timer.start(1000)

        # Status
        self.status_summary.setText("Estado: Grabando datos fisiológicos...")
        self.status_summary.setStyleSheet(f"color: {COLORS['red']}; padding: 8px; font-weight: bold;")

    def _end_session(self):
        # Confirm ending recording
        reply = QMessageBox.question(
            self, "Finalizar Grabación",
            "¿Seguro que desea finalizar la grabación y cosechar los datos?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.trigger.send("SESSION_END")
        self._log("⬛ Grabación finalizada (SESSION_END enviado).")

        # Temporarily disable UI during cooldown
        self.btn_end.setEnabled(False)
        self.status_summary.setText("Estado: Esperando escritura a disco (4s)...")
        self.status_summary.setStyleSheet(f"color: {COLORS['orange']}; padding: 8px;")

        # Capture subject ID
        id_numerico = str(self.spin_id.value()).zfill(3)
        nombre = self.nombre_input.text().strip()
        subject = f"{id_numerico}_{nombre}" if nombre else id_numerico

        # Wait 4 seconds for disk writing before harvesting
        QTimer.singleShot(4000, lambda: self._complete_session_end(subject))

    def _complete_session_end(self, subject):
        self.session_timer.stop()
        self.is_recording = False

        # Run harvest & data collection logic
        self._harvest_physio_data(subject)
        collected = self._collect_session_data(subject)
        unified_path = self._unify_data(subject)

        # Re-enable UI
        self.btn_launch.setEnabled(True)
        self.btn_end.setVisible(False)
        self.btn_next_subject.setVisible(True)
        self.spin_id.setEnabled(True)
        self.nombre_input.setEnabled(True)

        self.status_summary.setText("Estado: ✅ Datos cosechados y listos")
        self.status_summary.setStyleSheet(f"color: {COLORS['green']}; padding: 8px; font-weight: bold;")

        if unified_path:
            self._log(f"🧬 [Unificador] Archivos sincronizados en: {unified_path}")
            
        if collected:
            msg = QMessageBox(self)
            msg.setWindowTitle("Datos Cosechados")
            msg.setText(f"🏁 Grabación finalizada para el sujeto {subject}.\n\n"
                        f"📂 Datos recolectados en:\n{collected}")
            if unified_path:
                msg.setInformativeText(f"🧬 Archivo unificado de sincronía temporal creado:\n{unified_path}")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.exec()
        else:
            QMessageBox.information(
                self, "Datos Cosechados",
                f"🏁 Grabación finalizada para el sujeto {subject}.\n"
                "⚠️ Datos no recolectados automáticamente.\n"
                "Verifique manualmente las carpetas.")

    # ── Signal Handlers ───────────────────────────────────────────────────

    def _on_step_changed(self, idx: int, status: str):
        if 0 <= idx < len(self.step_widgets):
            self.step_widgets[idx].set_status(status)

    def _on_log(self, msg: str):
        self.log_console.append(msg)
        cursor = self.log_console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_console.setTextCursor(cursor)

    def _on_dialog_request(self, step_idx: int, phase: str):
        """Show memory question dialog — executes in main thread."""
        dlg = MemoryDialog(phase, self)
        dlg.exec()

    def _on_process_launched(self, name: str, pid: int):
        """Update system status when a process is launched."""
        # Track PID in the dashboard for cleanup on exit
        if pid not in self.active_processes:
            self.active_processes.append(pid)
            
        name_map = {
            "GazePointer": "GazePointer",
            "Monitor Fisiológico": "Monitor Fisiológico",
            "OGAMA": "OGAMA",
            "PsychoPy MMST": "PsychoPy MMST",
        }
        display = name_map.get(name, name)
        if display in self.sys_labels:
            dot, lbl = self.sys_labels[display]
            dot.setText("🟢")
            lbl.setText(f"PID {pid}")
            lbl.setStyleSheet(f"color: {COLORS['green']}; font-weight: bold;")

    def preparar_siguiente_sujeto(self):
        """
        ERD — Flujo de Transición: Siguiente Sujeto.
        Incrementa ID numérico automáticamente (+1), limpia nombre y restablece la UI
        para una nueva grabación sin reiniciar el software.
        """
        # 1. Detener transmisiones UDP residuales
        try:
            self.trigger.send("SESSION_RESET")
        except Exception:
            pass

        # 2. Incrementar ID numérico automáticamente (+1)
        self.spin_id.setValue(self.spin_id.value() + 1)

        # 3. Limpiar campo Nombre_Caracteres
        self.nombre_input.clear()
        self.nombre_input.setFocus()

        # 4. Habilitar UI para nueva grabación
        self.is_recording = False
        self.spin_id.setEnabled(True)
        self.nombre_input.setEnabled(True)
        self.btn_launch.setEnabled(True)
        self.btn_end.setVisible(False)
        self.btn_next_subject.setVisible(False)

        # 5. Restaurar step widgets al estado pendiente
        for sw in self.step_widgets:
            sw.set_status("pending")

        # 6. Resetear timer
        self._session_start = None
        self.timer_label.setText("00:00:00")
        self.timer_label.setStyleSheet(f"color: {COLORS['text_muted']};")

        self.status_summary.setText("Estado: Listo para nuevo sujeto")
        self.status_summary.setStyleSheet(f"color: {COLORS['text_dim']}; padding: 8px;")
        self._log(f"➡️ Sujeto preparado. ID N° = {self.spin_id.value():03d}. Ingrese nombre e inicie la grabación.")
        self.timer_label.setStyleSheet(f"color: {COLORS['text_dim']};")

    def _harvest_physio_data(self, subject_id: str):
        """
        Cosechadora de Datos — mueve CSVs del Monitor Fisiológico
        generados en la última hora a la carpeta del sujeto.
        """
        monitor_source_path = str(CFG["monitor_data_source"])
        monitor_source = Path(monitor_source_path)
        target_dir = DATASET_DIR / str(subject_id) / "Fisiologia"

        if not monitor_source.exists():
            self._log(f"⚠️ [Cosechadora] Origen no encontrado: {monitor_source}")
            return

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._log(f"❌ [Cosechadora] No se pudo crear carpeta destino: {e}")
            return

        self._log(f"🌾 [Cosechadora] Buscando CSVs en: {monitor_source_path}")
        now = time.time()
        moved = 0
        errors = 0

        for f in monitor_source.glob("*.csv"):
            try:
                age_seconds = now - f.stat().st_mtime
                if age_seconds < 3600:  # Modificado en la última hora
                    dest_file = target_dir / f.name
                    shutil.move(str(f), str(dest_file))
                    self._log(f"  ✅ {f.name} → Fisiologia/")
                    moved = moved + 1  # type: ignore
            except Exception as e:
                self._log(f"  ❌ Error moviendo {f.name}: {e}")
                errors = errors + 1  # type: ignore

        if moved > 0:
            self._log(f"🌾 [Cosechadora] {moved} archivo(s) movido(s) → {target_dir}")
        elif errors == 0:
            self._log("🌾 [Cosechadora] Sin CSVs recientes para mover (< 1h).")

    # ── Data Collection ───────────────────────────────────────────────────

    def _collect_session_data(self, subject_id: str) -> str:
        """
        Copy monitor CSVs and OGAMA experiment data into
        data/raw/{subject_id}/ for organized storage.
        Returns the destination path on success, empty string on failure.
        """
        dest_base_path = str(CFG["data_dest_base"])
        dest_base = Path(dest_base_path)
        dest = dest_base / str(subject_id)
        try:
            os.makedirs(dest, exist_ok=True)
        except Exception as e:
            self._log(f"❌ No se pudo crear carpeta destino: {e}")
            return ""

        files_copied = 0
        session_date = datetime.now().strftime("%Y%m%d")

        # ── 1. Monitor Fisiológico CSVs ───────────────────────
        monitor_src_path = str(CFG["monitor_data_source"])
        if os.path.isdir(monitor_src_path):
            self._log(f"📂 Buscando CSVs del monitor en: {monitor_src_path}")
            # Match CSVs containing the subject ID (case-insensitive)
            for csv_file in glob.glob(os.path.join(monitor_src_path, "PPG_GSR_*.csv")):
                csv_str = str(csv_file)
                basename = str(os.path.basename(csv_str)).lower()
                if subject_id.lower() in basename or session_date in basename:
                    dst_file = dest / "monitor" / Path(csv_str).name
                    os.makedirs(dst_file.parent, exist_ok=True)
                    shutil.copy2(csv_str, dst_file)
                    self._log(f"  ✅ {Path(csv_str).name} → monitor/")
                    files_copied = files_copied + 1  # type: ignore

            # Also grab the most recent timestamped session folder
            session_dirs = sorted(
                [d for d in Path(monitor_src_path).iterdir()
                 if d.is_dir() and d.name[0].isdigit()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if session_dirs:
                newest = session_dirs[0]
                dst_session = dest / "monitor" / newest.name
                if not dst_session.exists():
                    shutil.copytree(newest, dst_session)
                    self._log(f"  ✅ Sesión {newest.name} → monitor/")
                    files_copied = files_copied + 1  # type: ignore
        else:
            self._log(f"⚠️ Carpeta del monitor no encontrada: {monitor_src_path}")

        # ── 1.5. MMST Events CSVs ─────────────────────────────────
        mmst_src_path = str(BASE_DIR / "eventos_mmst")
        if os.path.isdir(mmst_src_path):
            self._log(f"📂 Buscando CSVs del MMST en: {mmst_src_path}")
            for csv_file in glob.glob(os.path.join(mmst_src_path, "*.csv")):
                csv_str = str(csv_file)
                basename = str(os.path.basename(csv_str)).lower()
                # Assuming id format like "008_Nestor_20260317"
                if subject_id.lower() in basename or session_date in basename:
                    dst_file = dest / "mmst" / Path(csv_str).name
                    os.makedirs(dst_file.parent, exist_ok=True)
                    shutil.copy2(csv_str, dst_file)
                    self._log(f"  ✅ {Path(csv_str).name} → mmst/")
                    files_copied = files_copied + 1

        # ── 2. OGAMA Experiments ──────────────────────────────
        ogama_src_path = str(CFG["ogama_data_source"])
        if os.path.isdir(ogama_src_path):
            self._log(f"📂 Buscando experimentos OGAMA en: {ogama_src_path}")
            ogama_dirs = sorted(
                [d for d in Path(ogama_src_path).iterdir() if d.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if ogama_dirs:
                newest_ogama = ogama_dirs[0]
                dst_ogama = dest / "ogama" / newest_ogama.name
                if not dst_ogama.exists():
                    shutil.copytree(newest_ogama, dst_ogama)
                    self._log(f"  ✅ {newest_ogama.name} → ogama/")
                    files_copied = files_copied + 1  # type: ignore
        else:
            self._log(f"⚠️ Carpeta de OGAMA no encontrada: {ogama_src_path}")

        # ── 3. Session log copy ───────────────────────────────
        session_log = LOGS_DIR / "session.log"
        if session_log.exists():
            shutil.copy2(session_log, dest / f"session_{session_date}.log")
            self._log("  ✅ session.log copiado")
            files_copied = files_copied + 1  # type: ignore

        self._log(f"📦 Recolección completa: {files_copied} elementos → {dest}")
        return str(dest) if files_copied > 0 else ""

    def _unify_data(self, subject_id: str) -> str:
        """
        Lee el CSV fisiológico y el CSV del MMST, y los cruza basándose en el
        timestamp para encontrar la respuesta fisiológica concurrente al estímulo visual.
        Crea un archivo unificado en la carpeta `Datos_Unificados`.
        """
        import pandas as pd
        dest_base_path = str(CFG["data_dest_base"])
        dest = Path(dest_base_path) / str(subject_id)
        
        monitor_dir = dest / "monitor"
        mmst_dir = dest / "mmst"
        unified_dir = dest / "Datos_Unificados"
        
        if not monitor_dir.exists() or not mmst_dir.exists():
            self._log("⚠️ No se puede unificar datos: Faltan carpetas de monitor o mmst.")
            return ""
            
        monitor_csvs = list(monitor_dir.glob("PPG_GSR_*.csv"))
        mmst_csvs = list(mmst_dir.glob("*.csv"))
        
        if not monitor_csvs or not mmst_csvs:
            self._log("⚠️ No se puede unificar datos: Archivos CSV faltantes.")
            return ""
            
        try:
            # Sort to get latest
            monitor_csvs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            mmst_csvs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            p_file = monitor_csvs[0]
            m_file = mmst_csvs[0]
            
            # Read files skipping metadata rows (assuming comments start with #)
            df_phys = pd.read_csv(p_file, comment='#')
            df_mmst = pd.read_csv(m_file, comment='#')
            
            if df_phys.empty or df_mmst.empty:
                self._log("⚠️ Unificación fallida: Uno de los DataFrames está vacío.")
                return ""
            
            # Sort on timestamp
            df_phys = df_phys.sort_values(by="Timestamp")
            df_mmst = df_mmst.sort_values(by="timestamp_onset")
            
            # Using merge_asof to align event times to closest preceding/succeeding physiological timestamp
            # We want to match each MMST visual event with the closest available physiological state
            unified_df = pd.merge_asof(
                df_mmst,
                df_phys,
                left_on="timestamp_onset",
                right_on="Timestamp",
                direction="nearest",
                tolerance=2.0 # Allow max 2s diff just in case
            )
            
            unified_dir.mkdir(parents=True, exist_ok=True)
            output_file = unified_dir / f"Unified_Data_{subject_id}_{datetime.now().strftime('%Y%m%d')}.csv"
            unified_df.to_csv(output_file, index=False)
            
            return str(output_file)
            
        except Exception as e:
            self._log(f"❌ Error durante unificación de datos: {str(e)}")
            return ""

    # ── Manual Override Managed Launchers ─────────────────────────────────

    def _execute_managed_manual_task(self, label: str, exe_path: str, args=None, priority='normal'):
        """Lanza una tarea manual supervisada por un hilo independiente."""
        if not os.path.exists(exe_path) and not exe_path.endswith('.py'):
            self._log(f"❌ No se encontró: {exe_path}")
            QMessageBox.warning(self, "Recurso no encontrado", f"No se pudo localizar:\n{exe_path}")
            return

        # Guard anti-duplicación para Monitor y MMST
        if label in self.sys_labels:
            dot, _ = self.sys_labels[label]
            if dot.text() == "🟢":
                self._log(f"⚠️ {label} ya se encuentra activo.")
                return

        managed = ManagedTaskThread(label, exe_path, args=args, priority=priority)
        
        # Conexiones de señales
        managed.log_message.connect(self._log)
        managed.task_started.connect(self._on_process_launched)
        managed.task_finished.connect(lambda name, rc: self._on_task_finished_manual(name, rc))
        
        # Almacenar referencia para evitar GC y permitir cierre limpio
        if not hasattr(self, '_managed_pool'):
            self._managed_pool = {}
        self._managed_pool[label] = managed
        
        managed.start()
        self._log(f"⚙️ Iniciando {label} en hilo supervisado...")

    def _on_task_finished_manual(self, name: str, rc: int):
        """Actualiza la UI al finalizar una tarea manual."""
        if name in self.sys_labels:
            dot, lbl = self.sys_labels[name]
            dot.setText("⏸️")
            lbl.setText("IDLE")
            lbl.setStyleSheet(f"color: {COLORS['text_muted']};")
        self._log(f"ℹ️ {name} ha finalizado (Exit code: {rc})")

    def _force_launch_monitor(self):
        self._execute_managed_manual_task("Monitor Fisiológico", CFG["monitor_exe"], priority='above_normal')

    def _force_launch_gazepointer(self):
        self._execute_managed_manual_task("GazePointer", CFG["gazepointer_exe"])

    def _force_launch_ogama(self):
        self._execute_managed_manual_task("OGAMA", CFG["ogama_exe"])

    def _force_compile_mmst(self):
        self._execute_managed_manual_task("Compilador MMST", CFG["mmst_script"], args=["--compile"])

    def _force_launch_mmst(self):
        id_numerico = str(self.spin_id.value()).zfill(3)
        nombre = self.nombre_input.text().strip()
        subject_id = f"{id_numerico}_{nombre}" if nombre else id_numerico
        self._execute_managed_manual_task("PsychoPy MMST", CFG["mmst_script"], args=[subject_id], priority='below_normal')

    def _force_launch_relaxation(self):
        script = str(BASE_DIR / "src" / "relaxation_task.py")
        if not os.path.exists(script):
             script = str(Path(__file__).parent / "relaxation_task.py")
        self._execute_managed_manual_task("Relajación", script, priority='below_normal')

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Asegurar que el cursor reaparezca si la app se cierra bruscamente
        if hasattr(self, 'cursor_hider'):
            self.cursor_hider.abort()
            self.cursor_hider.wait(1000)

        if self.is_recording:
            reply = QMessageBox.question(
                self, "Cerrar",
                "La grabación está actualmente activa.\n¿Desea cerrar de todas formas?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            # Se detiene el timer y se envia el reset de sesion si se decide cerrar
            try:
                self.trigger.send("SESSION_END")
            except:
                pass
            

        # Tidy up all tracked processes (from both thread and manual overrides)
        if hasattr(self, 'active_processes') and self.active_processes:
            self._log("🧹 Limpiando procesos activos antes de salir...")
            for pid in self.active_processes:
                try:
                    # Usar taskkill en Windows para forzar el cierre de procesos hijos también
                    creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                                  capture_output=True, creationflags=creation_flags)
                except Exception:
                    pass
            self.active_processes.clear()
            
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def apply_dark_palette(app: QApplication):
    """Apply a cohesive dark Fusion palette."""
    app.setStyle("Fusion")
    palette = QPalette()

    palette.setColor(QPalette.ColorRole.Window,          QColor(COLORS["bg_dark"]))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base,             QColor(COLORS["bg_panel"]))
    palette.setColor(QPalette.ColorRole.AlternateBase,    QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase,      QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.ColorRole.ToolTipText,      QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Text,             QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button,           QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.ColorRole.ButtonText,       QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.BrightText,       QColor(COLORS["red"]))
    palette.setColor(QPalette.ColorRole.Highlight,        QColor(COLORS["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText,  QColor(COLORS["bg_dark"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText,  QColor(COLORS["text_muted"]))

    # Disabled
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(COLORS["text_muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(COLORS["text_muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(COLORS["text_muted"]))

    app.setPalette(palette)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_palette(app)

    # Typography
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Global stylesheet
    app.setStyleSheet(GLOBAL_STYLE)

    window = MissionControlDashboard()
    window.show()
    sys.exit(app.exec())