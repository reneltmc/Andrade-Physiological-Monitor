#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Andrade Physiological Analysis Suite Pro - PyQt6 Edition
---------------------------------------------------------
Advanced physiological signal analysis and stress detection system.
Native PyQt6 implementation for high-performance data visualization.
"""

import sys
import os
import traceback
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")  # Forzar backend de Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection

from scipy import stats as sp_stats
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# PyQt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QDockWidget, QScrollArea, QTabWidget, 
                             QMdiArea, QMdiSubWindow, QPushButton, QLabel, 
                             QCheckBox, QSpinBox, QComboBox, QDoubleSpinBox, 
                             QGroupBox, QFileDialog, QMessageBox, QTableWidget, 
                             QTableWidgetItem, QAbstractItemView, QFormLayout, 
                             QListWidget, QStyleFactory, QSplitter, QLineEdit, QInputDialog)
from PyQt6.QtCore import Qt, QTimer, QSettings, QSize
from PyQt6.QtGui import QIcon, QAction, QPixmap
import ctypes
import pickle
import json

def calculate_d_prime(hits, misses, false_alarms, correct_rejections):
    """
    Calcula el índice d' (sensibilidad) y c (sesgo de respuesta) 
    para pruebas de memoria declarativa.
    Aplica corrección log-lineal para tasas extremas (0 o 1).
    """
    # Total de estímulos (Target = mostrados previamente, Noise = nuevos)
    total_targets = hits + misses
    total_noise = false_alarms + correct_rejections
    
    if total_targets == 0 or total_noise == 0:
        return np.nan, np.nan
        
    # Cálculo de tasas
    hit_rate = hits / total_targets
    fa_rate = false_alarms / total_noise
    
    # Corrección para tasas extremas (evita Z = infinito)
    # Se ajusta utilizando la regla de 1/(2N) recomendada en la teoría de detección de señales
    if hit_rate == 1.0:
        hit_rate = 1.0 - (1.0 / (2.0 * total_targets))
    elif hit_rate == 0.0:
        hit_rate = 1.0 / (2.0 * total_targets)
        
    if fa_rate == 1.0:
        fa_rate = 1.0 - (1.0 / (2.0 * total_noise))
    elif fa_rate == 0.0:
        fa_rate = 1.0 / (2.0 * total_noise)
        
    # Transformación Z (inversa de la función de distribución acumulativa normal estándar)
    z_hit = sp_stats.norm.ppf(hit_rate)
    z_fa = sp_stats.norm.ppf(fa_rate)
    
    # Índice de sensibilidad d'
    d_prime = z_hit - z_fa
    
    # Sesgo de respuesta c (valores negativos indican sesgo liberal, positivos conservador)
    c_bias = -(z_hit + z_fa) / 2.0
    
    return float(d_prime), float(c_bias)

# HAS_SKLEARN será determinado perezosamente

from PyQt6.QtWidgets import QFrame, QSizeGrip

class DashboardArea(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Widget)
        dummy = QWidget()
        dummy.setFixedSize(0, 0)
        self.setCentralWidget(dummy)
        self.setDockOptions(QMainWindow.DockOption.AllowNestedDocks | 
                            QMainWindow.DockOption.AnimatedDocks)
        self.dock_widgets = []

    def add_signal_panel(self, title, color_hex, widget_content):
        dock = QDockWidget(title, self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | 
                         QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setStyleSheet(f"""
            QDockWidget {{
                border: 2px solid {color_hex};
                color: #cdd6f4;
                font-weight: bold;
            }}
            QDockWidget::title {{
                background: #313150;
                padding-left: 10px;
                padding-top: 4px;
            }}
        """)
        dock.setWidget(widget_content)
        
        # Secuencia cascada para el docking autoestibado:
        if not self.dock_widgets:
            self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
        else:
            # Dividir con el panel anterior para apilamiento vertical automático
            self.splitDockWidget(self.dock_widgets[-1], dock, Qt.Orientation.Vertical)
        
        self.dock_widgets.append(dock)

        # -------------------------------------------------------------------
        # FIX SCROLL: Forzar tamaño mínimo del Main Window interno porque
        # Qt asume que su tamaño útil es 0, lo que causa el colapso.
        # -------------------------------------------------------------------
        total_min_h = sum(d.widget().minimumHeight() for d in self.dock_widgets if d.widget())
        # Añade un extra para los títulos de cada Dock (aprox 30px) y márgenes (5px)
        self.setMinimumHeight(total_min_h + (len(self.dock_widgets) * 35))
        
        return dock

    def clear_docks(self):
        for dock in self.dock_widgets:
            self.removeDockWidget(dock)
            dock.deleteLater()
        self.dock_widgets.clear()

class ResizablePlotWidget(QFrame):
    def __init__(self, fig, title, bg_color, border_color, text_color, parent=None, initial_h=350):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet(f"QFrame {{ background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 6px; }}")
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 20, 20)
        
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(f"color: {text_color}; font-weight: bold; border: none; font-size: 14px;")
        self.layout.addWidget(self.title_lbl)
        
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("border: none; background: transparent;")
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas, stretch=1)
        
        self.grip = QSizeGrip(self)
        
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.setMinimumSize(150, initial_h) # Ancho reducido para ajuste de pantalla, alto respeta initial_h
        
        # Conexión para el zoom con la rueda del ratón
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None: return
        
        # Factor de escala base
        base_scale = 1.15
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        # Coordenadas actuales del ratón
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None: return

        # Recalcular límites matemáticos focalizados hacia el cursor
        cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.canvas.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        rect = self.rect()
        self.grip.move(rect.right() - self.grip.width(), rect.bottom() - self.grip.height())
        self.canvas.draw_idle()


class StaticPlotWidget(QFrame):
    def __init__(self, fig, title, bg_color, border_color, text_color, parent=None, initial_h=350):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet(f"QFrame {{ background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 6px; }}")
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(f"color: {text_color}; font-weight: bold; border: none; font-size: 14px;")
        self.layout.addWidget(self.title_lbl)
        
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("border: none; background: transparent;")
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas, stretch=1)
        
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.setMinimumSize(150, initial_h) # Ancho reducido para ajuste de pantalla, alto respeta initial_h


# ═══════════════════════════════════════════════════════════════════════════════
# THEME ENGINE & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CLASSROOM_DARK = {
    "bg":         "#202124", "surface":    "#292A2D", "surface2":   "#3C4043",
    "border":     "#5F6368", "text":       "#E8EAED", "subtext":    "#9AA0A6",
    "accent":     "#8AB4F8", "green":      "#81C995", "red":        "#F28B82",
    "peach":      "#FCAD70", "yellow":     "#FDE293", "mauve":      "#D7AEFB",
    "teal":       "#78D9EC", "pink":       "#FF8BCB",
}

CLASSROOM_LIGHT = {
    "bg":         "#F8F9FA", "surface":    "#FFFFFF", "surface2":   "#F1F3F4",
    "border":     "#DADCE0", "text":       "#3C4043", "subtext":    "#5F6368",
    "accent":     "#1A73E8", "green":      "#1E8E3E", "red":        "#D93025",
    "peach":      "#FA7B17", "yellow":     "#F9AB00", "mauve":      "#9334E6",
    "teal":       "#12B5CB", "pink":       "#E52592",
}

DARK_THEME = CLASSROOM_DARK
LIGHT_THEME = CLASSROOM_LIGHT

PALETTE_KEYS = ["accent", "green", "red", "peach", "mauve", "teal", "yellow", "pink"]

KNOWN_VARIABLES_META = [
    {"canonical": "heart_rate_bpm", "unit": "BPM", "color_key": "red", "label": "Ritmo Cardíaco HR (BPM)", "synonyms": ["heart_rate_bpm", "heart_rate", "hr", "bpm", "pulse"]},
    {"canonical": "rmssd_ms", "unit": "ms", "color_key": "green", "label": "Variabilidad RMSSD (ms)", "synonyms": ["rmssd_ms", "rmssd"]},
    {"canonical": "sdnn_ms", "unit": "ms", "color_key": "peach", "label": "Variabilidad SDNN (ms)", "synonyms": ["sdnn_ms", "sdnn"]},
    {"canonical": "ln_rmssd", "unit": "Ln(ms)", "color_key": "green", "label": "Ln(RMSSD) Normalizado", "synonyms": ["ln_rmssd", "log_rmssd"]},
    {"canonical": "ln_sdnn", "unit": "Ln(ms)", "color_key": "peach", "label": "Ln(SDNN) Normalizado", "synonyms": ["ln_sdnn", "log_sdnn"]},
    {"canonical": "conductance_us", "unit": "µS", "color_key": "accent", "label": "Conductancia SCL (µS)", "synonyms": ["conductance_us", "conductance", "scl", "gsr"]},
    {"canonical": "scr_component", "unit": "µS", "color_key": "mauve", "label": "Componente SCR (µS)", "synonyms": ["scr_component", "scr", "phasic"]},
    {"canonical": "resistance_ohm", "unit": "kΩ", "color_key": "yellow", "label": "Resistencia Piel (kΩ)", "synonyms": ["resistance_ohm", "resistance"]},
    {"canonical": "gsr_raw", "unit": "Raw", "color_key": "teal", "label": "GSR Señal Raw", "synonyms": ["gsr_raw"]},
    {"canonical": "signal_quality", "unit": "%", "color_key": "pink", "label": "Calidad Señal (%)", "synonyms": ["signal_quality", "quality"]},
    {"canonical": "scl_zscore", "unit": "z", "color_key": "teal", "label": "SCL (Z-Score)", "synonyms": ["scl_zscore", "scl_z"]},
    {"canonical": "scr_zscore", "unit": "z", "color_key": "pink", "label": "SCR (Z-Score)", "synonyms": ["scr_zscore", "scr_z"]},
]

STRESS_CONFIG = {
    "z_score_threshold": 2.0, "scr_min_height": 0.02, "window_seconds": 30,
    "stress_min_duration": 5, "stress_merge_gap": 10, "hr_elevation_threshold": 1.5,
    "scr_spike_threshold": 2.0, "rmssd_drop_threshold": -1.5, "sdnn_drop_threshold": -1.5,
    "baseline_percentile": 25, "smoothing_window": 5,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AcuteStressDetector:
    def __init__(self, config=None):
        self.config = config or STRESS_CONFIG
        self.stress_episodes = []

    def detect_stress(self, data, time_col="time_rel"):
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or (isinstance(data, dict) and not data): return []
        
        if isinstance(data, dict):
            t_min = min(t[0] for t, *_ in data.values() if len(t) > 0)
            t_max = max(t[-1] for t, *_ in data.values() if len(t) > 0)
            if np.isnan(t_min) or np.isnan(t_max): return []
            common_t = np.arange(t_min, t_max, 1.0)
            df = pd.DataFrame({time_col: common_t})
            for col, (t_arr, y_arr, *_) in data.items():
                mask = ~np.isnan(y_arr)
                if len(t_arr[mask]) > 1:
                    df[col] = np.interp(common_t, t_arr[mask], y_arr[mask], left=np.nan, right=np.nan)
        else:
            df = data

        time = df[time_col].values
        n = len(time)
        scores = {k: np.zeros(n) for k in ("hr", "scr", "eda", "rmssd", "sdnn")}
        avail = []

        if "heart_rate_bpm" in df.columns:
            scores["hr"] = self._z_elevation(df["heart_rate_bpm"].values, self.config["hr_elevation_threshold"]); avail.append("HR")
        if "scr_component" in df.columns:
            scores["scr"] = self._scr_spikes(df["scr_component"].values); avail.append("SCR")
        if "conductance_us" in df.columns:
            scores["eda"] = self._z_elevation(df["conductance_us"].values, self.config["scr_spike_threshold"]); avail.append("EDA")
        if "rmssd_ms" in df.columns:
            scores["rmssd"] = self._z_decrease(df["rmssd_ms"].values, self.config["rmssd_drop_threshold"]); avail.append("RMSSD")

        if not avail: return []

        weights = {"hr": 1.0, "scr": 1.2, "eda": 1.0, "rmssd": 0.8, "sdnn": 0.8}
        combined = np.zeros(n)
        tw = sum(weights[k] for k, s in scores.items() if np.any(s > 0))
        for k, s in scores.items():
            if np.any(s > 0): combined += s * weights[k]
        if tw > 0: combined /= tw

        stress_scores = uniform_filter1d(combined, size=self.config["smoothing_window"])
        self.stress_episodes = self._extract_episodes(time, stress_scores, 0.35, self.config["stress_min_duration"], self.config["stress_merge_gap"])
        return self.stress_episodes

    def _z_score(self, data):
        d = np.asarray(data, dtype=float)
        bl = np.nanpercentile(d, self.config["baseline_percentile"])
        sd = np.nanstd(d)
        if sd < 1e-6: return np.zeros_like(d)
        return np.nan_to_num((d - bl) / sd, nan=0.0)

    def _z_elevation(self, data, threshold): return uniform_filter1d(np.clip(self._z_score(data) / threshold, 0, 1), size=3)
    def _z_decrease(self, data, threshold): return uniform_filter1d(np.clip(-self._z_score(data) / abs(threshold), 0, 1), size=5)
    def _scr_spikes(self, data):
        d = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
        peaks, _ = find_peaks(d, height=self.config["scr_min_height"], distance=30, prominence=0.01)
        score = np.zeros_like(d)
        for p in peaks:
            lo, hi = max(0, p - 15), min(len(score), p + 15)
            score[lo:hi] = np.maximum(score[lo:hi], d[p])
        mx = np.max(score)
        return score / mx if mx > 0 else score

    def _extract_episodes(self, time, score, thr, min_dur, merge_gap):
        above = score > thr
        eps, in_ep, si = [], False, 0
        for i, a in enumerate(above):
            if a and not in_ep: si, in_ep = i, True
            elif not a and in_ep:
                if time[i] - time[si] >= min_dur: eps.append((time[si], time[i], float(np.mean(score[si:i]))))
                in_ep = False
        if in_ep and time[-1] - time[si] >= min_dur: eps.append((time[si], time[-1], float(np.mean(score[si:]))))
        
        if len(eps) > 1:
            merged = [eps[0]]
            for ep in eps[1:]:
                last = merged[-1]
                if ep[0] - last[1] < merge_gap: merged[-1] = (last[0], ep[1], (last[2] + ep[2]) / 2)
                else: merged.append(ep)
            eps = merged
        return eps

    @staticmethod
    def stress_color(intensity, colors_dict):
        if intensity > 0.75: return colors_dict["red"]
        elif intensity > 0.5: return colors_dict["peach"]
        return colors_dict["yellow"]


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)
        
        self.btn = QPushButton(f"▼ {title}")
        self.btn.setCheckable(True)
        self.btn.setChecked(True)
        self.btn.setStyleSheet("QPushButton { text-align: left; font-weight: bold; padding: 6px; border: none; background-color: #313150; color: #89b4fa; }")
        self.btn.clicked.connect(self.toggle)
        self.lay.addWidget(self.btn)
        
        self.content = QFrame()
        self.content.setFrameShape(QFrame.Shape.NoFrame)
        self.content_lay = QVBoxLayout(self.content)
        self.content_lay.setContentsMargins(10, 5, 0, 5)
        self.lay.addWidget(self.content)
        
    def toggle(self):
        is_visible = self.content.isVisible()
        self.content.setVisible(not is_visible)
        self.btn.setText(f"{'▶' if is_visible else '▼'} {self.btn.text()[2:]}")
        
    def layout(self):
        return self.content_lay

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION (PyQt6)
# ═══════════════════════════════════════════════════════════════════════════════

class PhysioAnalyzerPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhysioAnalyzer Pro")
        self.resize(1300, 850)

        # Configuración de Icono del Programa
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.logo_icon_path = os.path.join(script_dir, "logo.ico")
        self.logo_png_path = os.path.join(script_dir, "logo.png")
        
        if os.path.exists(self.logo_icon_path):
            self.setWindowIcon(QIcon(self.logo_icon_path))
        elif os.path.exists(self.logo_png_path):
            self.setWindowIcon(QIcon(self.logo_png_path))

        self.is_dark = True
        self.colors = DARK_THEME

        # Atributos de Datos
        self.dfs = {}
        self.project_signals = {}  # Nuevo almacén de señales originales: {sheet_name: {var_name: (t_arr, y_arr)}}
        self.file_origins = {}
        self.active_sheet = None
        self.df_full = None
        self.df = None
        self.signals = {}          # Señales activas locales
        self.file_path = None
        self.force_resample = False # Forzar remuestreo común (Hz)
        self.available_cols = []
        self.var_checks = {}
        self.var_group_checks = {}
        self.font_size_tabs = 14
        self.custom_colors = {} # Dict para guardar el mapeo color->columna/sujeto
        self.tab_selected_color = None # Color personalizado para tabs
        self.drag_active = False # Arrastrar umbral
        self.threshold_y = None
        self.erd_mappings = {}  # Mapeo Sujeto Fisiológico -> Log MMST
        self.excluded_signals = {} # NUEVO: {sheet_name: ['heart_rate_bpm']}
        self.custom_intervals = {} # Intervalos de tiempo personalizados por sujeto

        self.stress_detector = AcuteStressDetector()
        self.stress_episodes = []

        self.subject_phases = {} # {sheet_name: {'pre': (0.0, 100.0), 'post': (100.0, 250.0)}}
        
        # Estados de Controles
        self.show_stress = True
        self.multi_normalize_var = True
        self.grp_is_paired = True
        self.grp_normalize_var = False
        self.mmst_opacity = 0.15
        self.mmst_color_aversiva = "#f38ba8"
        self.mmst_color_neutra = "#a6e3a1"
        self.gantt_mode = True
        self.hide_physio_monitor = False

        self.time_min = 0.0
        self.time_max = 1.0
        
        # Buffers e Historial
        self.delete_history = []
        self.picker_data = {}
        self.fig_cache = None
        self.delete_mode = False
        
        # Inicialización de referencias a listas de sujetos (Pre-instanciadas para evitar errores de sincronización)
        self.list_compare_sheets = QListWidget()
        self.list_phases_sheets = QListWidget()
        self.list_g1 = QListWidget()
        self.list_g2 = QListWidget()

        self.log_path = os.path.join(os.getcwd(), "signal_debug.txt")
        self.settings = QSettings("AndradePhysio", "PhysioAnalyzerPro")
        self.project_path = None

        self._init_ui()
        self._setup_timers()
        self._apply_theme()
        
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Listo. Carga un archivo o inicia sesión...", 5000)
        
    def showEvent(self, event):
        """Aparece una vez que la ventana es visible para evitar bloqueos de inicialización."""
        super().showEvent(event)
        if not hasattr(self, '_app_started'):
             self._app_started = True
             # Paso 1: Recuperación de Desastres (Cache)
             QTimer.singleShot(500, self._check_session_recovery)
             # Paso 2: Preguntar por Sesión / Proyecto al inicio
             QTimer.singleShot(1500, self._prompt_initial_session)

    def _init_ui(self):
        # 1. Barra de Herramientas Superior / Menu
        self._create_top_bar()

        # 2. Panel Lateral de Controles (QDockWidget)
        self._create_sidebar()

        # 3. Contenedor Central (QTabWidget)
        self.notebook = QTabWidget()
        self.setCentralWidget(self.notebook)

        # Crear Pestañas y sus Contenedores Docking
        self.plot_hosts = {}
        tabs_config = [
            ("📊 Señales", "signals", None),
            ("📈 Descriptivos", "descr", None),
            ("🔔 Distribución", "dist", None),
            ("📉 Regresión", "reg", self._create_reg_controls),
            ("🧩 Clusters", "clust", self._create_clust_controls),
            ("📊 Comparar Hojas", "compare", self._create_compare_controls),
            ("👥 Análisis Grupal", "group", self._create_group_controls),
            ("🎬 Análisis ERD", "erd", self._create_erd_controls),
            ("⚖️ Comparar Fases", "phases", self._create_phases_controls),
            ("🔮 Multivariado", "multivar", self._create_multivar_controls)
        ]

        for label, key, control_builder in tabs_config:
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.setSpacing(2)

            # Insertar controles específicos si los hay
            if control_builder:
                ctrl_bar = control_builder()
                if ctrl_bar:
                    from PyQt6.QtWidgets import QSizePolicy
                    # Evitar que la barra se expanda verticalmente ocupando espacio de las gráficas
                    ctrl_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                    
                    ctrl_scroll = QScrollArea()
                    ctrl_scroll.setWidget(ctrl_bar)
                    ctrl_scroll.setWidgetResizable(True)
                    ctrl_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                    ctrl_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
                    
                    if key in ['compare', 'group', 'phases', 'multivar']:
                        ctrl_scroll.setFixedHeight(110)
                    else:
                        ctrl_scroll.setFixedHeight(60)
                        
                    tab_layout.addWidget(ctrl_scroll)

            # --- SISTEMA UNIFICADO DE DASHBOARD (DashboardArea) ---
            # Aplicamos el sistema de DashboardArea flotante/estibables a todas las pestañas
            # para garantizar que el motor de docks de Qt mantenga las geometrías estables y visibles.
            scroll_area = QScrollArea(tab_widget)
            scroll_area.setWidgetResizable(True)
            scroll_area.setStyleSheet("QScrollArea { border: none; }")
            
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            scroll_layout.setContentsMargins(0, 0, 0, 0)
            
            dashboard = DashboardArea()
            scroll_layout.addWidget(dashboard)
            
            scroll_area.setWidget(scroll_content)
            # Priorizar que el scroll_area absorba todo el espacio vertical sobrante (stretch=1)
            tab_layout.addWidget(scroll_area, stretch=1)
            self.plot_hosts[key] = dashboard

            self.notebook.addTab(tab_widget, label)

        self.notebook.currentChanged.connect(self._refresh_current_tab)

    def _create_top_bar(self):
        # --- MENÚ SUPERIOR (Nativo PyQt6) ---
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&Archivo")
        
        act_open = QAction("📂 Abrir Proyecto (.physio)", self)
        act_open.triggered.connect(self.load_project)
        file_menu.addAction(act_open)
        
        act_save = QAction("💾 Guardar Proyecto", self)
        act_save.triggered.connect(self.save_project)
        file_menu.addAction(act_save)
        
        file_menu.addSeparator()
        self.menu_recent = file_menu.addMenu("🕒 Proyectos Recientes")
        self._rebuild_file_menu()
        
        file_menu.addSeparator()
        act_exit = QAction("❌ Salir", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # --- BARRA DE HERRAMIENTAS (Toolbar) ---
        toolbar = self.addToolBar("Controles Globales")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(32, 32))
        
        # Logo de Branding en Toolbar
        if os.path.exists(self.logo_png_path):
            logo_branding = QLabel()
            logo_branding.setPixmap(QPixmap(self.logo_png_path).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            logo_branding.setStyleSheet("margin-right: 10px; margin-left: 5px;")
            toolbar.addWidget(logo_branding)
        
        # Botones de Proyecto en Toolbar
        btn_save_proj = QPushButton("💾 Guardar Proyecto")
        btn_save_proj.setStyleSheet("background-color: #313150; font-weight: bold;")
        btn_save_proj.clicked.connect(self.save_project)
        toolbar.addWidget(btn_save_proj)
        
        btn_load_proj = QPushButton("📂 Abrir Proyecto")
        btn_load_proj.clicked.connect(self.load_project)
        toolbar.addWidget(btn_load_proj)
        
        toolbar.addSeparator()
        
        btn_load = QPushButton("➕ Cargar CSV/Excel")
        btn_load.clicked.connect(self.load_file)
        toolbar.addWidget(btn_load)

        btn_mmst = QPushButton("🎬 Cargar Log MMST")
        btn_mmst.clicked.connect(self.load_mmst_log)
        toolbar.addWidget(btn_mmst)

        btn_siad = QPushButton("📜 Cargar Log SIAD")
        btn_siad.clicked.connect(self.load_siad_log)
        toolbar.addWidget(btn_siad)
        
        # --- NUEVO BOTÓN: Exportación LMM (R) ---
        btn_export_lmm = QPushButton("📈 Exportar LMM (R)")
        btn_export_lmm.setStyleSheet("background-color: #1E8E3E; font-weight: bold; color: white;")
        btn_export_lmm.clicked.connect(self._export_lmm_r_format)
        toolbar.addWidget(btn_export_lmm)
        # ----------------------------------------
        
        btn_clear_intervals = QPushButton("🧹")
        btn_clear_intervals.setToolTip("Limpiar Intervalos")
        btn_clear_intervals.clicked.connect(self.clear_custom_intervals)
        toolbar.addWidget(btn_clear_intervals)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("   📊 Sujeto Actual: "))
        self.cmb_sheet = QComboBox()
        self.cmb_sheet.setMinimumWidth(200)
        self.cmb_sheet.currentTextChanged.connect(self._on_sheet_change)
        toolbar.addWidget(self.cmb_sheet)

        from PyQt6.QtWidgets import QSizePolicy
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        toolbar.addWidget(spacer)
        btn_theme = QPushButton("🌓 Tema")
        btn_theme.clicked.connect(self.toggle_theme)
        toolbar.addWidget(btn_theme)

        btn_png = QPushButton("💾 Guardar PNG")
        btn_png.clicked.connect(self.export_image)
        toolbar.addWidget(btn_png)

    def _create_sidebar(self):
        self.dock = QDockWidget("⚙ Controles", self)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # --- Información del Archivo Activo ---
        self.lbl_file = QLabel(" 📄 Ningún archivo cargado")
        self.lbl_file.setStyleSheet(f"font-weight: bold; color: {self.colors['subtext']}; padding: 5px;")
        self.lbl_file.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_file)

        # --- Grupo 1: Variables Individuales (Plegable) ---
        self.var_box = CollapsibleBox("Variables Individuales")
        self.var_layout = self.var_box.layout()
        sidebar_layout.addWidget(self.var_box)

        # --- Grupo 2: Variables de Análisis Grupal (Plegable, oculto por defecto) ---
        self.group_vars_container = CollapsibleBox("Variables de Análisis Grupal")
        self.var_group_layout = self.group_vars_container.layout()
        self.group_vars_container.setVisible(False)
        sidebar_layout.addWidget(self.group_vars_container)

        # --- Auxiliares ---
        chk_stress = QCheckBox("🔍 Detección Estrés Agudo")
        chk_stress.setChecked(self.show_stress)
        chk_stress.toggled.connect(self._toggle_stress_detect)
        sidebar_layout.addWidget(chk_stress)

        self.chk_force_resample = QCheckBox("🔄 Forzar remuestreo común (Hz: 60)")
        self.chk_force_resample.setChecked(self.force_resample)
        self.chk_force_resample.toggled.connect(self._toggle_force_resample)
        sidebar_layout.addWidget(self.chk_force_resample)

        # Opciones de Intervalos SIAD
        self.chk_gantt_mode = QCheckBox("📊 Mostrar eventos como barras (Gantt)")
        self.chk_gantt_mode.setChecked(self.gantt_mode)
        self.chk_gantt_mode.toggled.connect(self._toggle_gantt_mode)
        sidebar_layout.addWidget(self.chk_gantt_mode)
        
        self.chk_hide_physio = QCheckBox("🙈 Ocultar Monitor Fisiológico")
        self.chk_hide_physio.setChecked(self.hide_physio_monitor)
        self.chk_hide_physio.toggled.connect(self._toggle_hide_physio)
        sidebar_layout.addWidget(self.chk_hide_physio)

        sidebar_layout.addWidget(QLabel("Rango Temporal (Global):"))
        f_range = QWidget()
        f_range_layout = QHBoxLayout(f_range)
        f_range_layout.setContentsMargins(0, 0, 0, 0)
        
        self.spn_start = QDoubleSpinBox()
        self.spn_start.setRange(0, 99999)
        self.spn_start.valueChanged.connect(self._apply_range)
        f_range_layout.addWidget(QLabel("Inicio:"))
        f_range_layout.addWidget(self.spn_start)

        self.spn_end = QDoubleSpinBox()
        self.spn_end.setRange(0, 99999)
        self.spn_end.setValue(100.0)
        self.spn_end.valueChanged.connect(self._apply_range)
        f_range_layout.addWidget(QLabel("Fin:"))
        f_range_layout.addWidget(self.spn_end)
        
        sidebar_layout.addWidget(f_range)

        # Opacidad MMST
        sidebar_layout.addWidget(QLabel("Superposición MMST Opacidad:"))
        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0, 1.0)
        self.spn_alpha.setSingleStep(0.05)
        self.spn_alpha.setValue(self.mmst_opacity)
        self.spn_alpha.valueChanged.connect(self._update_opacity)
        sidebar_layout.addWidget(self.spn_alpha)

        # Edición datos
        self.chk_delete_mode = QCheckBox("🎯 Eliminar Puntos (Pick)")
        self.chk_delete_mode.toggled.connect(self._toggle_delete_mode)
        sidebar_layout.addWidget(self.chk_delete_mode)

        self.btn_undo = QPushButton("↩ Deshacer Borrado")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self._undo_delete)
        sidebar_layout.addWidget(self.btn_undo)

        # --- Grupo 3: Guillotina de Outliers (Umbral) ---
        self.grp_threshold = QGroupBox("✂ Guillotina de Outliers")
        l_thresh = QVBoxLayout(self.grp_threshold)
        
        f_thresh_var = QHBoxLayout()
        f_thresh_var.addWidget(QLabel("Variable:"))
        self.cmb_thresh_var = QComboBox()
        self.cmb_thresh_var.addItems(["heart_rate_bpm", "conductance_us", "scr_component", "gsr_raw"])
        f_thresh_var.addWidget(self.cmb_thresh_var)
        l_thresh.addLayout(f_thresh_var)
        
        self.chk_threshold_line = QCheckBox("Activar Línea de Umbral (Drag)")
        self.chk_threshold_line.toggled.connect(self._refresh_current_tab)
        l_thresh.addWidget(self.chk_threshold_line)
        
        # NUEVO: Botón para invalidar variable
        self.btn_invalidate_var = QPushButton("🚫 Purgar Variable (Drop-out)")
        self.btn_invalidate_var.setStyleSheet("background-color: #5c2b29; color: #f38ba8;")
        self.btn_invalidate_var.clicked.connect(self._invalidate_subject_variable)
        l_thresh.addWidget(self.btn_invalidate_var)
        
        sidebar_layout.addWidget(self.grp_threshold)

        # Controles de Personalización (Aparecen solo en Modo Claro)
        self.grp_custom_ui = QGroupBox("🎨 Personalización Visual")
        self.grp_custom_ui.setVisible(not self.is_dark)
        l_ui = QVBoxLayout(self.grp_custom_ui)
        
        # Tamaño Pestañas
        f_font = QHBoxLayout()
        f_font.addWidget(QLabel("Tamaño Fuente Tabs:"))
        self.spn_tab_font = QSpinBox()
        self.spn_tab_font.setRange(8, 16)
        self.spn_tab_font.setValue(self.font_size_tabs)
        self.spn_tab_font.valueChanged.connect(self._apply_custom_ui)
        f_font.addWidget(self.spn_tab_font)
        l_ui.addLayout(f_font)
        
        # Color Variable / Ventana en viñeta
        f_pick_var = QHBoxLayout()
        f_pick_var.addWidget(QLabel("Elemento:"))
        self.cmb_custom_target = QComboBox()
        self.cmb_custom_target.addItems(["Pestañas Generales"])
        f_pick_var.addWidget(self.cmb_custom_target)
        l_ui.addLayout(f_pick_var)

        self.btn_pick_color = QPushButton("🖌 Cambiar Color")
        self.btn_pick_color.clicked.connect(self._pick_variable_color)
        l_ui.addWidget(self.btn_pick_color)

        sidebar_layout.addWidget(self.grp_custom_ui)

        scroll_area.setWidget(sidebar_widget)
        self.dock.setWidget(scroll_area)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)

    def _apply_custom_ui(self):
        self.font_size_tabs = self.spn_tab_font.value()
        self._apply_theme()

    def _pick_variable_color(self):
        from PyQt6.QtWidgets import QColorDialog
        if self.is_dark: return
        target = self.cmb_custom_target.currentText()
        if not target: return
        
        color = QColorDialog.getColor(title=f"Color para {target}")
        if color.isValid():
            hex_color = color.name()
            if target == "Pestañas Generales":
                self.tab_selected_color = hex_color
                self._apply_theme()
            else:
                self.custom_colors[target] = hex_color
                self._refresh_current_tab()

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-Builders para Controles de Tab
    # ─────────────────────────────────────────────────────────────────────────
    def _create_reg_controls(self):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)
        l.addWidget(QLabel("Variable X:"))
        self.cmb_reg_x = QComboBox()
        self.cmb_reg_x.currentTextChanged.connect(self._plot_regression)
        l.addWidget(self.cmb_reg_x)

        l.addWidget(QLabel("Variable Y:"))
        self.cmb_reg_y = QComboBox()
        self.cmb_reg_y.currentTextChanged.connect(self._plot_regression)
        l.addWidget(self.cmb_reg_y)
        l.addStretch()
        return w

    def _create_clust_controls(self):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)
        l.addWidget(QLabel("Var X:"))
        self.cmb_cl_x = QComboBox()
        self.cmb_cl_x.currentTextChanged.connect(self._plot_clusters)
        l.addWidget(self.cmb_cl_x)

        l.addWidget(QLabel("Var Y:"))
        self.cmb_cl_y = QComboBox()
        self.cmb_cl_y.currentTextChanged.connect(self._plot_clusters)
        l.addWidget(self.cmb_cl_y)

        l.addWidget(QLabel("K clusters:"))
        self.spn_k = QSpinBox()
        self.spn_k.setRange(2, 10)
        self.spn_k.setValue(3)
        self.spn_k.valueChanged.connect(self._plot_clusters)
        l.addWidget(self.spn_k)
        l.addStretch()
        return w

    def _create_compare_controls(self):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)
        
        # self.list_compare_sheets ya está instanciada en __init__
        self.list_compare_sheets.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_compare_sheets.itemSelectionChanged.connect(self._refresh_current_tab)
        self.list_compare_sheets.setMaximumHeight(80)
        l.addWidget(QLabel("Series:"))
        l.addWidget(self.list_compare_sheets)

        self.cmb_compare_var = QComboBox()
        self.cmb_compare_var.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(QLabel("Variable:"))
        l.addWidget(self.cmb_compare_var)

        self.chk_multi_normalize = QCheckBox("Δ React. Neta")
        self.chk_multi_normalize.setChecked(self.multi_normalize_var)
        self.chk_multi_normalize.toggled.connect(self._toggle_multi_normalize)
        l.addWidget(self.chk_multi_normalize)

        self.cmb_compare_type = QComboBox()
        self.cmb_compare_type.addItems(["Evolución Temporal", "Estadística Descriptiva", "Distribución Densidad"])
        self.cmb_compare_type.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(self.cmb_compare_type)

        btn_export_compare = QPushButton("💾 Exportar Datos")
        btn_export_compare.clicked.connect(self._export_compare_data)
        l.addWidget(btn_export_compare)

        # Controles de intervalo temporal
        l.addWidget(QLabel("<b>🕒 T min:</b>"))
        self.comp_tmin_var = QDoubleSpinBox()
        self.comp_tmin_var.setRange(0, 99999)
        self.comp_tmin_var.valueChanged.connect(self._refresh_current_tab)
        l.addWidget(self.comp_tmin_var)

        l.addWidget(QLabel("<b>🕒 T max:</b>"))
        self.comp_tmax_var = QDoubleSpinBox()
        self.comp_tmax_var.setRange(0, 99999)
        self.comp_tmax_var.setValue(300.0)
        self.comp_tmax_var.valueChanged.connect(self._refresh_current_tab)
        l.addWidget(self.comp_tmax_var)

        return w

    def _create_group_controls(self):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)

        # Listado Grupo A (ya instanciado)
        self.list_g1.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_g1.itemSelectionChanged.connect(self._refresh_current_tab)
        self.list_g1.setMaximumHeight(80)
        l.addWidget(QLabel("Grupo A:"))
        l.addWidget(self.list_g1)

        # Listado Grupo B (ya instanciado)
        self.list_g2.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_g2.itemSelectionChanged.connect(self._refresh_current_tab)
        self.list_g2.setMaximumHeight(80)
        l.addWidget(QLabel("Grupo B:"))
        l.addWidget(self.list_g2)

        # Labels fijos para grupos

        # Combo Variables
        self.grp_var_cmb = QComboBox()
        self.grp_var_cmb.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(QLabel("Var:"))
        l.addWidget(self.grp_var_cmb)

        # Métrica
        self.grp_metric_cmb = QComboBox()
        self.grp_metric_cmb.addItems(["Media", "Mediana", "Máximo", "Mínimo", "Desv.Est", "Varianza"])
        self.grp_metric_cmb.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(QLabel("Métrica:"))
        l.addWidget(self.grp_metric_cmb)

        # Caja is_paired
        self.grp_is_paired = QCheckBox("Pareado (Pre/Post)")
        self.grp_is_paired.setChecked(True)
        self.grp_is_paired.toggled.connect(self._refresh_current_tab)
        l.addWidget(self.grp_is_paired)

        self.lbl_stat_suggestion = QLabel("<b>Sugerencia:</b> Pendiente")
        self.lbl_stat_suggestion.setStyleSheet("color: #81C995;")
        l.addWidget(self.lbl_stat_suggestion)

        # Rango Pre
        self.grp_tmin_var = QDoubleSpinBox()
        self.grp_tmin_var.setRange(0, 99999); self.grp_tmin_var.valueChanged.connect(self._refresh_current_tab)
        self.grp_tmax_var = QDoubleSpinBox()
        self.grp_tmax_var.setRange(0, 99999); self.grp_tmax_var.setValue(100); self.grp_tmax_var.valueChanged.connect(self._refresh_current_tab)
        l.addWidget(QLabel("Pre-Min:"))
        l.addWidget(self.grp_tmin_var)
        l.addWidget(QLabel("Pre-Max:"))
        l.addWidget(self.grp_tmax_var)

        # Rango Post
        self.grp_tmin_post_var = QDoubleSpinBox()
        self.grp_tmin_post_var.setRange(0, 99999); self.grp_tmin_post_var.valueChanged.connect(self._refresh_current_tab)
        self.grp_tmax_post_var = QDoubleSpinBox()
        self.grp_tmax_post_var.setRange(0, 99999); self.grp_tmax_post_var.setValue(320); self.grp_tmax_post_var.valueChanged.connect(self._refresh_current_tab)
        l.addWidget(QLabel("Post-Min:"))
        l.addWidget(self.grp_tmin_post_var)
        l.addWidget(QLabel("Post-Max:"))
        l.addWidget(self.grp_tmax_post_var)

        l.addStretch()
        return w

    def _create_phases_controls(self):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)

        self.list_phases_sheets.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_phases_sheets.itemSelectionChanged.connect(self._on_phases_list_select_change)
        self.list_phases_sheets.setMaximumHeight(160)
        l.addWidget(QLabel("<b>Sujetos:</b>"))
        l.addWidget(self.list_phases_sheets)
        
        l.addWidget(QLabel("<b>🔵 Comparación de Fases:</b>"))
        
        from PyQt6.QtWidgets import QLineEdit
        l.addWidget(QLabel("Fase 1:"))
        self.cmb_ph1_src = QComboBox()
        self.cmb_ph1_src.addItems(["Relajación", "MMST", "OGAMA 1", "OGAMA 2", "Manual 1", "Manual 2"])
        self.cmb_ph1_src.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(self.cmb_ph1_src)
        self.le_ph1_name = QLineEdit("Fase 1")
        self.le_ph1_name.setFixedWidth(70)
        self.le_ph1_name.textChanged.connect(self._refresh_current_tab)
        l.addWidget(self.le_ph1_name)
        self.ph1_start = QDoubleSpinBox(); self.ph1_start.setRange(0, 100000); self.ph1_start.setDecimals(1)
        self.ph1_end = QDoubleSpinBox(); self.ph1_end.setRange(0, 100000); self.ph1_end.setDecimals(1)
        self.ph1_start.valueChanged.connect(self._sync_phases_to_dict)
        self.ph1_end.valueChanged.connect(self._sync_phases_to_dict)
        l.addWidget(QLabel("De:"))
        l.addWidget(self.ph1_start)
        l.addWidget(QLabel("a:"))
        l.addWidget(self.ph1_end)
        
        l.addWidget(QLabel("Fase 2:"))
        self.cmb_ph2_src = QComboBox()
        self.cmb_ph2_src.addItems(["Relajación", "MMST", "OGAMA 1", "OGAMA 2", "Manual 1", "Manual 2"])
        self.cmb_ph2_src.setCurrentText("MMST")
        self.cmb_ph2_src.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(self.cmb_ph2_src)
        self.le_ph2_name = QLineEdit("Fase 2")
        self.le_ph2_name.setFixedWidth(70)
        self.le_ph2_name.textChanged.connect(self._refresh_current_tab)
        l.addWidget(self.le_ph2_name)
        self.ph2_start = QDoubleSpinBox(); self.ph2_start.setRange(0, 100000); self.ph2_start.setDecimals(1)
        self.ph2_end = QDoubleSpinBox(); self.ph2_end.setRange(0, 100000); self.ph2_end.setDecimals(1)
        self.ph2_start.valueChanged.connect(self._sync_phases_to_dict)
        self.ph2_end.valueChanged.connect(self._sync_phases_to_dict)
        l.addWidget(QLabel("De:"))
        l.addWidget(self.ph2_start)
        l.addWidget(QLabel("a:"))
        l.addWidget(self.ph2_end)
        
        self.cmb_ph1_event = QComboBox()
        self.cmb_ph2_event = QComboBox()
        btn_apply_all = QPushButton("👥 Aplicar A Todos")
        btn_apply_all.clicked.connect(self._apply_phases_to_all)
        l.addWidget(btn_apply_all)

        l.addSpacing(15)
        l.addWidget(QLabel("<b>Variable:</b>"))
        self.cmb_phases_var = QComboBox()
        self.cmb_phases_var.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(self.cmb_phases_var)

        l.addSpacing(15)
        l.addWidget(QLabel("<b>⚙️ Test:</b>"))
        self.cmb_phases_test = QComboBox()
        self.cmb_phases_test.addItems([
            "Automática (Sugerida)", 
            "T-Student Pareada", 
            "Wilcoxon (No Paramétrica)", 
            "Modelo Lineal Mixto (LMM)"
        ])
        self.cmb_phases_test.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(self.cmb_phases_test)
        
        l.addSpacing(15)
        l.addWidget(QLabel("<b>α (Significancia):</b>"))
        self.cmb_phases_alpha = QComboBox()
        self.cmb_phases_alpha.addItems(['0.01 (1%)', '0.05 (5%)', '0.10 (10%)'])
        self.cmb_phases_alpha.setCurrentIndex(1)
        self.cmb_phases_alpha.currentTextChanged.connect(self._refresh_current_tab)
        l.addWidget(self.cmb_phases_alpha)

        l.addSpacing(15)
        
        l.addWidget(QLabel("<b>Tamaño Título:</b>"))
        self.spin_lmm_title_size = QSpinBox()
        self.spin_lmm_title_size.setRange(5, 40)
        self.spin_lmm_title_size.setValue(13)
        self.spin_lmm_title_size.setFixedWidth(50)
        self.spin_lmm_title_size.valueChanged.connect(self._refresh_current_tab)
        l.addWidget(self.spin_lmm_title_size)

        l.addWidget(QLabel("<b>P-valor Y:</b>"))
        self.spin_lmm_pval_offset = QDoubleSpinBox()
        self.spin_lmm_pval_offset.setRange(-2.0, 2.0)
        self.spin_lmm_pval_offset.setSingleStep(0.01)
        self.spin_lmm_pval_offset.setValue(0.05)
        self.spin_lmm_pval_offset.setFixedWidth(60)
        self.spin_lmm_pval_offset.valueChanged.connect(self._refresh_current_tab)
        l.addWidget(self.spin_lmm_pval_offset)

        l.addSpacing(15)
        
        # Nuevos controles interactivos para Comparar Fases
        self.phases_outliers_removed = set()
        
        self.chk_show_gantt = QCheckBox("Mostrar Diagrama de Gantt")
        self.chk_show_gantt.setChecked(True)
        self.chk_show_gantt.toggled.connect(self._refresh_current_tab)
        l.addWidget(self.chk_show_gantt)
        
        self.chk_clean_phases_outliers = QCheckBox("Limpiar Outliers (Clic)")
        self.chk_clean_phases_outliers.setToolTip("Habilita hacer clic en los puntos del diagrama para eliminar temporalmente a ese sujeto del análisis.")
        l.addWidget(self.chk_clean_phases_outliers)
        
        self.btn_reset_outliers = QPushButton("↺ Restaurar Outliers")
        self.btn_reset_outliers.clicked.connect(self._reset_phases_outliers)
        l.addWidget(self.btn_reset_outliers)

        l.addSpacing(15)

        l.addSpacing(15)
        btn_export_uni = QPushButton("💾 Exportar Datos")
        btn_export_uni.clicked.connect(self._export_univar_data)
        l.addWidget(btn_export_uni)

        l.addStretch()
        return w

    def _reset_phases_outliers(self):
        if hasattr(self, 'phases_outliers_removed') and self.phases_outliers_removed:
            self.phases_outliers_removed.clear()
            self._refresh_current_tab()

    def _create_multivar_controls(self):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)

        self.list_mv_sheets = QListWidget()
        self.list_mv_sheets.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_mv_sheets.setMaximumHeight(80)
        l.addWidget(QLabel("<b>Sujetos:</b>"))
        l.addWidget(self.list_mv_sheets)

        self.list_mv_vars = QListWidget()
        self.list_mv_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_mv_vars.setMaximumHeight(80)
        l.addWidget(QLabel("<b>Variables (Mín 2):</b>"))
        l.addWidget(self.list_mv_vars)

        ctrl_layout = QVBoxLayout()
        self.cmb_mv_method = QComboBox()
        self.cmb_mv_method.addItems([
            "PCA (Componentes Principales)", 
            "MANOVA (Fase 1 vs Fase 2)", 
            "Matriz de Correlación Parcial"
        ])
        ctrl_layout.addWidget(QLabel("<b>Método Multivariado:</b>"))
        ctrl_layout.addWidget(self.cmb_mv_method)

        self.chk_mv_zscore = QCheckBox("Estandarización Z-Score")
        self.chk_mv_zscore.setChecked(True)
        ctrl_layout.addWidget(self.chk_mv_zscore)

        btn_run = QPushButton("⚡ Generar Análisis")
        btn_run.setStyleSheet("font-weight: bold;")
        btn_run.clicked.connect(self._refresh_current_tab)
        ctrl_layout.addWidget(btn_run)

        btn_export = QPushButton("💾 Exportar Datos")
        btn_export.clicked.connect(self._export_multivar_data)
        ctrl_layout.addWidget(btn_export)

        l.addLayout(ctrl_layout)
        l.addStretch()
        return w

    def _export_multivar_data(self):
        if not hasattr(self, 'last_df_mv') or self.last_df_mv is None or self.last_df_mv.empty:
            QMessageBox.warning(self, "Sin datos", "No hay datos generados para exportar. Haz clic en 'Generar Análisis' primero.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Exportar Datos Multivariados", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not path:
            return
            
        try:
            if path.endswith('.csv'):
                self.last_df_mv.to_csv(path, index=False, encoding='utf-8')
            elif path.endswith('.xlsx'):
                self.last_df_mv.to_excel(path, index=False)
            QMessageBox.information(self, "Éxito", f"Datos exportados correctamente a:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Exportar", f"No se pudo guardar el archivo:\n{e}")

    def _export_univar_data(self):
        if not hasattr(self, 'last_df_uni') or self.last_df_uni is None or self.last_df_uni.empty:
            QMessageBox.warning(self, "Sin datos", "No hay datos generados para exportar. Haz clic en una variable primero.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Exportar Datos Univariados", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not path:
            return
            
        try:
            if path.endswith('.csv'):
                self.last_df_uni.to_csv(path, index=False, encoding='utf-8')
            elif path.endswith('.xlsx'):
                self.last_df_uni.to_excel(path, index=False)
            QMessageBox.information(self, "Éxito", f"Datos exportados correctamente a:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Exportar", f"No se pudo guardar el archivo:\n{e}")

    def _export_compare_data(self):
        if not hasattr(self, 'last_df_compare') or self.last_df_compare is None or self.last_df_compare.empty:
            QMessageBox.warning(self, "Sin datos", "No hay datos generados para exportar. Selecciona la vista de 'Estadística Descriptiva' primero.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Exportar Datos Descriptivos", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not path:
            return
            
        try:
            if path.endswith('.csv'):
                self.last_df_compare.to_csv(path, index=False, encoding='utf-8')
            elif path.endswith('.xlsx'):
                self.last_df_compare.to_excel(path, index=False)
            QMessageBox.information(self, "Éxito", f"Datos exportados correctamente a:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Exportar", f"No se pudo guardar el archivo:\n{e}")

    def _export_lmm_r_format(self):
        """
        Exporta los datos fisiológicos en formato largo (Tidy Data) 
        estrictamente formateado para modelos lineales mixtos en R (lme4).
        """
        if not self.dfs:
            QMessageBox.warning(self, "Sin datos", "No hay bases de datos cargadas para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Exportar Dataset LMM (R)", "Datos_Longitudinales_LMM.csv", "CSV Files (*.csv)")
        if not path:
            return

        # Variables fisiológicas solicitadas para el análisis LMM
        target_vars = ["ln_rmssd", "ln_sdnn", "heart_rate_bpm", "conductance_us", "scl_zscore"]
        rows = []

        try:
            for sheet, df_sub in self.dfs.items():
                # Extraer las fases definidas para el sujeto; si no existen, usar baseline global
                phases = self.subject_phases.get(sheet, {})
                if not phases:
                    phases = {'Baseline_Global': (float(df_sub['time_rel'].min()), float(df_sub['time_rel'].max()))}

                for phase_name, (t_start, t_end) in phases.items():
                    row_data = {
                        "Sujeto_ID": sheet,
                        "Fase": phase_name
                    }
                    
                    # Filtrar la matriz de datos por la ventana de tiempo estricta
                    mask = (df_sub["time_rel"] >= t_start) & (df_sub["time_rel"] <= t_end)
                    df_phase = df_sub.loc[mask]

                    for var in target_vars:
                        if df_phase.empty:
                            row_data[var] = np.nan
                            continue
                            
                        # Si la variable ya está calculada, extraer la media
                        if var in df_phase.columns:
                            row_data[var] = df_phase[var].mean()
                        # Generación dinámica de variables logarítmicas si no se guardaron en la limpieza
                        elif var == "ln_rmssd" and "rmssd_ms" in df_phase.columns:
                            valid_vals = df_phase["rmssd_ms"].replace(0, np.nan).dropna()
                            row_data[var] = np.log(valid_vals).mean() if not valid_vals.empty else np.nan
                        elif var == "ln_sdnn" and "sdnn_ms" in df_phase.columns:
                            valid_vals = df_phase["sdnn_ms"].replace(0, np.nan).dropna()
                            row_data[var] = np.log(valid_vals).mean() if not valid_vals.empty else np.nan
                        else:
                            row_data[var] = np.nan
                            
                    rows.append(row_data)

            # Estructuración final y exportación a CSV
            df_lmm = pd.DataFrame(rows)
            df_lmm.to_csv(path, index=False, encoding='utf-8')
            
            QMessageBox.information(self, "Exportación Exitosa", 
                                    f"Matriz LMM generada correctamente en:\n{path}\n\n"
                                    "Estructura lista para R:\n"
                                    "lmer(variable ~ Fase + (1|Sujeto_ID), data=df)")

        except Exception as e:
            QMessageBox.critical(self, "Error al Exportar LMM", f"Se produjo un error durante la extracción de datos:\n{str(e)}")

    def _sync_phases_to_dict(self):
        sel = self.list_phases_sheets.selectedItems()
        targets = [item.text() for item in sel] if sel else ([self.active_sheet] if self.active_sheet else [])
        if not targets: return
        
        for target in targets:
            if target not in self.subject_phases:
                 self.subject_phases[target] = {}
            self.subject_phases[target]['Manual 1'] = (self.ph1_start.value(), self.ph1_end.value())
            self.subject_phases[target]['Manual 2'] = (self.ph2_start.value(), self.ph2_end.value())
        self._refresh_current_tab()

    def _apply_phases_to_all(self):
        if not hasattr(self, 'active_sheet') or not self.active_sheet: return
        p1_s, p1_e = self.ph1_start.value(), self.ph1_end.value()
        p2_s, p2_e = self.ph2_start.value(), self.ph2_end.value()
        for sheet in self.dfs.keys():
            if sheet not in self.subject_phases:
                 self.subject_phases[sheet] = {}
            self.subject_phases[sheet]['Manual 1'] = (p1_s, p1_e)
            self.subject_phases[sheet]['Manual 2'] = (p2_s, p2_e)
        if hasattr(self, 'statusBar') and self.statusBar:
            self.statusBar.showMessage("✅ Rangos manuales aplicados a todos los sujetos", 3000)

    def _on_phases_list_select_change(self):
        # Actualizar SpinBoxes con el último seleccionado si hay uno solo para guiar al usuario
        sel = self.list_phases_sheets.selectedItems()
        if sel:
            sheet_name = sel[-1].text()
            if sheet_name in self.subject_phases:
                p = self.subject_phases[sheet_name]
                self.ph1_start.blockSignals(True); self.ph1_end.blockSignals(True)
                self.ph2_start.blockSignals(True); self.ph2_end.blockSignals(True)
                if 'Manual 1' in p:
                    self.ph1_start.setValue(p['Manual 1'][0]); self.ph1_end.setValue(p['Manual 1'][1])
                if 'Manual 2' in p:
                    self.ph2_start.setValue(p['Manual 2'][0]); self.ph2_end.setValue(p['Manual 2'][1])
                self.ph1_start.blockSignals(False); self.ph1_end.blockSignals(False)
                self.ph2_start.blockSignals(False); self.ph2_end.blockSignals(False)
        self._refresh_current_tab()

    def _create_erd_controls(self):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)
        
        l.addWidget(QLabel("<b>Fuente Eventos:</b>"))
        self.cmb_erd_source_type = QComboBox()
        self.cmb_erd_source_type.addItems(["Log MMST"])
        self.cmb_erd_source_type.currentTextChanged.connect(self._update_erd_combo_source)
        l.addWidget(self.cmb_erd_source_type)

        l.addWidget(QLabel("<b>Archivo de Eventos:</b>"))
        self.cmb_erd_mmst = QComboBox()
        self.cmb_erd_mmst.setMinimumWidth(200)
        if hasattr(self, 'mmst_dfs'):
             self.cmb_erd_mmst.addItems(list(self.mmst_dfs.keys()))
        l.addWidget(self.cmb_erd_mmst)
        
        btn_link = QPushButton("🔗 Vincular")
        btn_link.clicked.connect(self._link_erd_subject)
        l.addWidget(btn_link)
        
        self.chk_erd_normalize = QCheckBox("Normalizar Reactividades (Baseline)")
        self.chk_erd_normalize.setChecked(True)
        self.chk_erd_normalize.toggled.connect(self._refresh_current_tab)
        l.addWidget(self.chk_erd_normalize)
        
        # Corrección manual de desfase temporal
        l.addWidget(QLabel("<b>Alineación Offset (s):</b>"))
        self.erd_offset_spn = QDoubleSpinBox()
        self.erd_offset_spn.setRange(-9999.0, 9999.0)
        self.erd_offset_spn.setValue(0.0)
        self.erd_offset_spn.setSingleStep(0.5)
        self.erd_offset_spn.valueChanged.connect(self._refresh_current_tab)
        l.addWidget(self.erd_offset_spn)
        
        l.addStretch()
        return w

    def _update_erd_combo_source(self):
        self.cmb_erd_mmst.clear()
        source_type = self.cmb_erd_source_type.currentText()
        if source_type == "Log MMST" and hasattr(self, 'mmst_dfs'):
            self.cmb_erd_mmst.addItems(list(self.mmst_dfs.keys()))
        self._refresh_current_tab()

    def _link_erd_subject(self):
        if not self.active_sheet: return
        selected_log = self.cmb_erd_mmst.currentText()
        if selected_log:
            self.erd_mappings[self.active_sheet] = selected_log
            if hasattr(self, 'statusBar') and self.statusBar is not None:
                self.statusBar.showMessage(f"Vínculo guardado: {self.active_sheet} ➔ {selected_log}", 3000)
            self._refresh_current_tab()

    def _setup_timers(self):
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._execute_refresh)

    # ─────────────────────────────────────────────────────────────────────────
    # DASHBOARD RENDERING COMPONENT
    # ─────────────────────────────────────────────────────────────────────────
    def _embed_figure(self, host_key: str, fig: Figure, title="Gráfica", force_clear=True, color_hex="#45456a"):
        import gc
        host_target = self.plot_hosts.get(host_key)
        if not host_target: return
        
        is_dashboard = hasattr(host_target, 'add_signal_panel')

        if force_clear:
             if is_dashboard:
                 host_target.clear_docks()
             else:
                 while host_target.count():
                     item = host_target.takeAt(0)
                     if item.widget():
                         item.widget().deleteLater()
             gc.collect() # Liberar memoria de figuras anteriores

        calculated_h = int(fig.get_figheight() * fig.dpi)
        if calculated_h < 350: calculated_h = 400

        if is_dashboard:
             # AQUÍ USAMOS ResizablePlotWidget (paneles flotantes)
             plot_widget = ResizablePlotWidget(
                 fig=fig, title=title, bg_color=self.colors["surface"], border_color=self.colors["accent"],
                 text_color="#ffffff" if self.is_dark else "#000000", initial_h=calculated_h, parent=None
             )
        else:
             # AQUÍ USAMOS StaticPlotWidget (Métricas pesadas dentro de QScrollArea)
             plot_widget = StaticPlotWidget(
                 fig=fig, title=title, bg_color=self.colors["surface"], border_color=self.colors["accent"],
                 text_color="#ffffff" if self.is_dark else "#000000", initial_h=calculated_h, parent=None
             )
        
        if is_dashboard:
            host_target.add_signal_panel(title, color_hex, plot_widget)
        else:
            host_target.addWidget(plot_widget)

        self.fig_cache = fig
        plot_widget.canvas.mpl_connect('pick_event', self._on_pick)
        
        try:
             fig.tight_layout()
        except:
             pass # Silenciar fallos de compresión geométrica extremos

        plot_widget.canvas.draw()
        plot_widget.show()
        
        # Guardar historial de vista inicial para Navigation Toolbar
        if hasattr(plot_widget, 'toolbar'):
            plot_widget.toolbar.push_current()
        
        # OBLIGAR a que Qt asimile la geometría física de este widget en el Layout:
        host_target.update()
        if is_dashboard:
            if hasattr(host_target, 'widget') and host_target.widget():
                 host_target.widget().adjustSize()
        else:
            # --- FIX CRITICO SCROLL COLLAPSE ---
            # Sumar las alturas mínimas de todos los widgets contenidos en el layout
            total_h = 0
            for i in range(host_target.count()):
                item = host_target.itemAt(i)
                if item and item.widget():
                    total_h += item.widget().minimumHeight() + host_target.spacing()
            
            p_widget = host_target.parentWidget()
            if p_widget:
                # Forzar un tamaño mínimo al scroll_content para que la ScrollArea se active
                p_widget.setMinimumHeight(total_h + 30)
                p_widget.adjustSize()
        
        # Evitar loops infinitos de redibujado durante la carga
        if getattr(self, '_refreshes_locked', False): return
        
        with open(self.log_path, "a") as f:
             f.write(f"--- WIDGET AÑADIDO: {title} | ALTURA ASIGNADA: {calculated_h}px | GEOMETRÍA FINAL: {plot_widget.geometry()} ---\n")
             f.flush()
        
        if not hasattr(self, '_keep_alive_widgets'): self._keep_alive_widgets = []
        if force_clear: self._keep_alive_widgets.clear()
        self._keep_alive_widgets.append(plot_widget)
        return plot_widget

    # ─────────────────────────────────────────────────────────────────────────
    # LÓGICA DE ACTUALIZACIÓN Y REDIBUJADO
    # ─────────────────────────────────────────────────────────────────────────
    def _refresh_current_tab(self):
        # Capturar qué está disparando el refresh para depuración
        import traceback
        stack = "".join(traceback.format_stack()[-5:])
        with open(self.log_path, "a") as f:
             f.write(f"\n[REFRESH TRIGGER] Source:\n{stack}\n")
             f.flush()

        if getattr(self, '_refreshes_locked', False): return
        self._redraw_timer.start(150)

    def _execute_refresh(self):
        if getattr(self, '_refreshing', False):
             print("[REFRESH] Bloqueada llamada concurrente para evitar colapso.")
             return
        self._refreshing = True
        
        idx = self.notebook.currentIndex()
        methods = [self._plot_signals, self._plot_descriptive, self._plot_distribution,
                   self._plot_regression, self._plot_clusters, self._plot_comparison, 
                   self._plot_group_stats, self._plot_erd, self._plot_phases, self._plot_multivariate]
        
        with open(self.log_path, "a") as f:
             f.write(f"--- REFRESH DISPARADO PARA TAB IDX: {idx} ---\n")
             f.flush()

        if idx < len(methods):
            try:
                methods[idx]()
                with open(self.log_path, "a") as f:
                     f.write(f"--- METODO DE TAB {idx} FINALIZÓ SIN ERRORES ---\n")
                     f.flush()
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(error_trace)
                if hasattr(self, 'statusBar'):
                    self.statusBar.showMessage(f"Error al generar gráfico: {e}", 8000)
                with open(self.log_path, "a") as f:
                     f.write(f"EXCEPTION IN TAB {idx}:\n{error_trace}\n")
            finally:
                self._refreshing = False

    def _toggle_stress_detect(self, checked):
        self.show_stress = checked
        self._refresh_current_tab()

    def _toggle_force_resample(self, checked):
        self.force_resample = checked
        if self.file_path and hasattr(self, 'active_sheet'):
            # Re-disparar limpieza de DataFrame y regenerar signals
            self._set_active_sheet(self.active_sheet)
        self._refresh_current_tab()

    def _toggle_multi_normalize(self, checked):
        self.multi_normalize_var = checked
        self._plot_comparison()

    def _toggle_gantt_mode(self, checked):
        self.gantt_mode = checked
        self._refresh_current_tab()

    def _toggle_hide_physio(self, checked):
        self.hide_physio_monitor = checked
        self._refresh_current_tab()

    def _toggle_delete_mode(self, checked):
        self.delete_mode = checked
        if hasattr(self, 'statusBar'):
            if checked:
                self.statusBar.showMessage("🎯 MODO BORRADO ACTIVADO: Haz clic en un punto de la gráfica para eliminarlo (Outlier).", 5000)
            else:
                self.statusBar.showMessage("Modo borrado desactivado.", 2000)

    def _update_opacity(self, val):
        self.mmst_opacity = val
        self._refresh_current_tab()

    def _apply_range(self):
        if self.force_resample or self.df_full is None:
            # Comportamiento antiguo (remuestreo o sin datos)
            t0, t1 = self.spn_start.value(), self.spn_end.value()
            if self.df_full is not None:
                self.df = self.df_full[(self.df_full["time_rel"] >= t0) & (self.df_full["time_rel"] <= t1)].copy()
                self.stress_episodes = self.stress_detector.detect_stress(self.df)
        else:
            # Filtrar señales originales por tiempo
            t0, t1 = self.spn_start.value(), self.spn_end.value()
            filtered_signals = {}
            source_signals = self.project_signals.get(self.active_sheet, {})
            for var, (t_arr, y_arr, idx_arr) in source_signals.items():
                mask = (t_arr >= t0) & (t_arr <= t1)
                filtered_signals[var] = (t_arr[mask], y_arr[mask], idx_arr[mask])
            self.signals = filtered_signals
            # El detector de estrés acepta diccionario de señales
            self.stress_episodes = self.stress_detector.detect_stress(self.signals)
        self._refresh_current_tab()

    def _active_vars(self):
        return [c for c, chk in self.var_checks.items() if chk.isChecked()]

    # ─────────────────────────────────────────────────────────────────────────
    # PLOTTING SKELETON (Adaptados para _embed_figure)
    # ─────────────────────────────────────────────────────────────────────────
    def _create_fig(self, rows=1, cols=1, **kwargs):
        w = kwargs.get('width', 10)
        h = kwargs.get('height_per', 3.5) * rows
        fig = Figure(figsize=(w, h), facecolor=self.colors["bg"])
        # Evitar colapsos silenciosos en PyQt6 Event Loop
        if rows * cols == 1:
             axes = [fig.add_subplot(111)]
        else:
             axes = fig.subplots(nrows=rows, ncols=cols)
             axes = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        
        # Aplicar estilo oscuro a los ejes para que sean visibles
        for ax in axes:
             ax.set_facecolor(self.colors["surface"])
             ax.tick_params(colors=self.colors["text"], labelsize=8)
             ax.xaxis.label.set_color(self.colors["text"])
             ax.yaxis.label.set_color(self.colors["text"])
             ax.title.set_color(self.colors["text"])
             for s_name, spine in ax.spines.items():
                  spine.set_color(self.colors["border"])
             # Ajustar Grid
             ax.grid(True, linestyle='--', color=self.colors["border"], alpha=0.2)
        return fig, axes
    def _var_meta(self, col):
        for v_info in KNOWN_VARIABLES_META:
            if col == v_info["canonical"]: 
                return v_info["unit"], self.colors[v_info["color_key"]], v_info["label"]
        return "Unidad", "#ffffff", col

    def _plot_signals(self):
        # Depuración
        with open(self.log_path, "w") as f:
             f.write(f"df is None: {self.df is None}\n")
             if self.df is not None:
                  f.write(f"df shape: {self.df.shape}\n")
                  f.write(f"df cols: {self.df.columns.tolist()}\n")
                  f.write(f"active_vars: {self._active_vars()}\n")
        if self.df is None:
            fig, axes = self._create_fig(1, 1)
            axes[0].text(0.5, 0.5, "Por favor, carga un archivo de datos (CSV/Excel).", 
                         ha="center", va="center", color=self.colors["subtext"], fontsize=12)
            axes[0].axis('off')
            self._embed_figure("signals", fig, title="Esperando Datos", force_clear=True)
            return

        cols = self._active_vars()
        self.picker_data = {}  
        
        if not cols:
            fig, axes = self._create_fig(1, 1)
            axes[0].text(0.5, 0.5, "Selecciona al menos una variable en el panel izquierdo.", 
                         ha="center", va="center", color=self.colors["subtext"], fontsize=12)
            axes[0].axis('off')
            self._embed_figure("signals", fig, title="Aviso", force_clear=True)
            return

        for i, col in enumerate(cols):
            with open(self.log_path, "a") as f:
                f.write(f"Iniciando loop para {col}...\n")
                f.flush()

            try:
                fig, axes = self._create_fig(rows=1, cols=1)
                ax = axes[0]
                
                yl, clr, title = self._var_meta(col)
                if self.signals and not self.force_resample:
                    time_arr, data_arr, index_arr = self.signals.get(col, (np.array([]), np.array([]), np.array([])))
                else:
                    time_arr = self.df["time_rel"].values
                    data_arr = self.df[col].values
                    index_arr = self.df.index.values

                if self.show_stress and self.stress_episodes:
                    for j, (s, e, inten) in enumerate(self.stress_episodes):
                        lbl = "Estrés Agudo" if j == 0 else None
                        ax.axvspan(s, e, color=self.stress_detector.stress_color(inten, self.colors), alpha=0.25, label=lbl)

                # --- Superposición de Eventos MMST ---
                try:
                    mapped_mmst_key = self.erd_mappings.get(self.active_sheet, self.active_sheet)
                    events_df = getattr(self, 'mmst_dfs', {}).get(mapped_mmst_key)
                    if (events_df is None or events_df.empty) and getattr(self, 'mmst_dfs', {}):
                         events_df = list(self.mmst_dfs.values())[0]  # Fallback
                    
                    # Fuzzy match fallback si hay múltiples logs y no coinciden las llaves exactas
                    if (events_df is None or events_df.empty) and hasattr(self, 'mmst_dfs'):
                         for k, df_ev in self.mmst_dfs.items():
                              if k.lower() in self.active_sheet.lower() or self.active_sheet.lower() in k.lower():
                                   events_df = df_ev
                                   break
                    
                    if events_df is not None and not events_df.empty:
                         onsets = events_df['timestamp_onset'].values
                         cats_raw = events_df['affective_category'].values if 'affective_category' in events_df.columns else np.array(['Unknown'] * len(onsets))
                         duration_ms = events_df['duration_ms'].values if 'duration_ms' in events_df.columns else np.array([5000.0] * len(onsets))
                         
                         # --- SAFETY CAP TO PREVENT HANGING ---
                         if len(onsets) > 500:
                              max_dur = np.nanmax(duration_ms) if len(duration_ms) > 0 else 0
                              threshold = 100 if max_dur > 100 else 0.1
                              valid_mask = (duration_ms > threshold) if 'duration_ms' in events_df.columns else np.ones(len(onsets), dtype=bool)
                              if np.any(valid_mask):
                                   onsets = onsets[valid_mask]
                                   cats_raw = cats_raw[valid_mask]
                                   duration_ms = duration_ms[valid_mask]
                              if len(onsets) > 500:
                                   onsets = onsets[:500]
                                   cats_raw = cats_raw[:500]
                                   duration_ms = duration_ms[:500]
                         
                         if 'timestamp' in self.df_full.columns:
                              num_ts = pd.to_numeric(self.df_full['timestamp'], errors='coerce')
                              abs_ts = num_ts[num_ts > 1e8]
                              t_sub_min_raw = abs_ts.min() if not abs_ts.empty else num_ts.min()
                              
                              if pd.isna(t_sub_min_raw):
                                   t_sub_min = self.df_full['time_rel'].min()
                              elif t_sub_min_raw > 1e14: t_sub_min = t_sub_min_raw / 1e6
                              elif t_sub_min_raw > 1e11: t_sub_min = t_sub_min_raw / 1000.0
                              else: t_sub_min = t_sub_min_raw
                         else:
                              t_sub_min = self.df_full['time_rel'].min()
                              
                         mx_onset = np.nanmax(onsets) if len(onsets) > 0 else 0
                         if mx_onset > 1e14: onset_sec_arr = onsets / 1e6
                         elif mx_onset > 1e11: onset_sec_arr = onsets / 1000.0
                         else: onset_sec_arr = onsets

                         if mx_onset > 86400 and t_sub_min < 86400:
                              onset_rel_arr = onset_sec_arr - np.nanmin(onset_sec_arr)
                         else:
                              onset_rel_arr = np.where(onset_sec_arr < 86400, onset_sec_arr, onset_sec_arr - t_sub_min)
                         
                         alpha_span = getattr(self, 'mmst_opacity', 0.15)
                         offset_s = getattr(self, 'erd_offset_spn', None).value() if hasattr(self, 'erd_offset_spn') else 0.0
                         
                         for j in range(len(onsets)):
                              onset_rel = onset_rel_arr[j] + offset_s
                              dur_sec = duration_ms[j] / 1000.0 if duration_ms[j] > 100 else 5.0
                              if duration_ms[j] < 50: dur_sec = 5.0 
                              
                              if np.isnan(onset_rel) or np.isnan(dur_sec): continue
                              # Evitar deformar el ejeX si el onset es gigantesco (fuera de rango)
                              if onset_rel > self.df_full['time_rel'].max() + 100: continue
                              
                              raw_cat = str(cats_raw[j]).strip().lower()
                              color = "#ffffff"
                              if 'aversiv' in raw_cat or 'negativ' in raw_cat: color = getattr(self, 'mmst_color_aversiva', '#f38ba8')
                              elif 'neutra' in raw_cat: color = getattr(self, 'mmst_color_neutra', '#a6e3a1')
                              elif 'pacifica' in raw_cat or 'positiv' in raw_cat: color = '#89b4fa'
                              
                              ax.axvspan(onset_rel, onset_rel + dur_sec, color=color, alpha=alpha_span, zorder=1)
                              ax.axvline(onset_rel, color=color, linestyle='--', alpha=0.3, zorder=2)
                except Exception as mmst_err:
                     print(f"Error superponiendo MMST: {mmst_err}")

                # --- Superposición de Eventos TXT ---
                try:
                    mapped_txt_key = self.erd_mappings.get(self.active_sheet, self.active_sheet)
                    txt_events_df = getattr(self, 'event_txt_dfs', {}).get(mapped_txt_key)
                    if (txt_events_df is None or txt_events_df.empty) and getattr(self, 'event_txt_dfs', {}):
                         txt_events_df = list(self.event_txt_dfs.values())[0]  # Fallback
                    
                    if (txt_events_df is None or txt_events_df.empty) and hasattr(self, 'event_txt_dfs'):
                         for k, df_ev in self.event_txt_dfs.items():
                              if k.lower() in self.active_sheet.lower() or self.active_sheet.lower() in k.lower():
                                   txt_events_df = df_ev
                                   break
                    
                    if txt_events_df is not None and not txt_events_df.empty:
                         onsets = txt_events_df['timestamp_onset'].values
                         cats_raw = txt_events_df['event_name'].values if 'event_name' in txt_events_df.columns else np.array(['Unknown'] * len(onsets))
                         duration_ms = txt_events_df['duration_ms'].values if 'duration_ms' in txt_events_df.columns else np.array([5000.0] * len(onsets))
                         
                         if len(onsets) > 500:
                              max_dur = np.nanmax(duration_ms) if len(duration_ms) > 0 else 0
                              threshold = 100 if max_dur > 100 else 0.1
                              valid_mask = (duration_ms > threshold) if 'duration_ms' in txt_events_df.columns else np.ones(len(onsets), dtype=bool)
                              if np.any(valid_mask):
                                   onsets = onsets[valid_mask]
                                   cats_raw = cats_raw[valid_mask]
                                   duration_ms = duration_ms[valid_mask]
                              if len(onsets) > 500:
                                   onsets = onsets[:500]
                                   cats_raw = cats_raw[:500]
                                   duration_ms = duration_ms[:500]
                         
                         if 'timestamp' in self.df_full.columns:
                              num_ts = pd.to_numeric(self.df_full['timestamp'], errors='coerce')
                              abs_ts = num_ts[num_ts > 1e8]
                              t_sub_min_raw = abs_ts.min() if not abs_ts.empty else num_ts.min()
                              if pd.isna(t_sub_min_raw): t_sub_min = self.df_full['time_rel'].min()
                              elif t_sub_min_raw > 1e14: t_sub_min = t_sub_min_raw / 1e6
                              elif t_sub_min_raw > 1e11: t_sub_min = t_sub_min_raw / 1000.0
                              else: t_sub_min = t_sub_min_raw
                         else:
                              t_sub_min = self.df_full['time_rel'].min()
                              
                         mx_onset = np.nanmax(onsets) if len(onsets) > 0 else 0
                         if mx_onset > 1e14: onset_sec_arr = onsets / 1e6
                         elif mx_onset > 1e11: onset_sec_arr = onsets / 1000.0
                         else: onset_sec_arr = onsets

                         if mx_onset > 86400 and t_sub_min < 86400:
                              onset_rel_arr = onset_sec_arr - np.nanmin(onset_sec_arr)
                         else:
                              onset_rel_arr = np.where(onset_sec_arr < 86400, onset_sec_arr, onset_sec_arr - t_sub_min)
                         
                         alpha_span = getattr(self, 'mmst_opacity', 0.15)
                         offset_s = getattr(self, 'erd_offset_spn', None).value() if hasattr(self, 'erd_offset_spn') else 0.0
                         
                         unique_cats = np.unique(cats_raw)
                         palette = [self.colors.get(k, '#ffffff') for k in ["accent", "peach", "yellow", "teal", "mauve", "pink"]]
                         cat_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)}

                         for j in range(len(onsets)):
                              onset_rel = onset_rel_arr[j] + offset_s
                              dur_sec = duration_ms[j] / 1000.0 if duration_ms[j] > 100 else 5.0
                              if duration_ms[j] < 50: dur_sec = 5.0 
                              
                              if np.isnan(onset_rel) or np.isnan(dur_sec): continue
                              if onset_rel > self.df_full['time_rel'].max() + 100: continue
                              
                              raw_cat = str(cats_raw[j])
                              color = cat_colors.get(raw_cat, '#89b4fa')
                              
                              ax.axvspan(onset_rel, onset_rel + dur_sec, color=color, alpha=alpha_span, zorder=1)
                              ax.axvline(onset_rel, color=color, linestyle=':', alpha=0.5, zorder=2)
                except Exception as txt_err:
                     print(f"Error superponiendo Eventos TXT: {txt_err}")

                # --- Superposición de Intervalos Personalizados ---
                subject_intervals = self.custom_intervals.get(self.active_sheet, []) if hasattr(self, 'custom_intervals') else []
                if subject_intervals:
                    try:
                         if 'timestamp' in self.df_full.columns:
                              num_ts = pd.to_numeric(self.df_full['timestamp'], errors='coerce')
                              abs_ts = num_ts[num_ts > 1e8]
                              t_sub_min_raw = abs_ts.min() if not abs_ts.empty else num_ts.min()
                              if pd.isna(t_sub_min_raw): t_sub_min = self.df_full['time_rel'].min()
                              elif t_sub_min_raw > 1e14: t_sub_min = t_sub_min_raw / 1e6
                              elif t_sub_min_raw > 1e11: t_sub_min = t_sub_min_raw / 1000.0
                              else: t_sub_min = t_sub_min_raw
                         else:
                              t_sub_min = self.df_full['time_rel'].min()

                         offset_s = getattr(self, 'erd_offset_spn', None).value() if hasattr(self, 'erd_offset_spn') else 0.0
                         unique_labels = set()
                         alpha_span = getattr(self, 'mmst_opacity', 0.15)
                         
                         gantt = getattr(self, 'gantt_mode', True)
                         t_start_view, t_end_view = self.spn_start.value(), self.spn_end.value()
                         
                         processed_intervals = []
                         for interval in subject_intervals:
                              lbl = interval['label']
                              
                              if getattr(self, 'hide_physio_monitor', False) and "monitor fisiológico" in lbl.lower():
                                  continue
                                  
                              s = interval['start']
                              e = interval['end']
                              color = interval['color']
                              
                              s_sec = s / 1e6 if s > 1e14 else (s / 1000.0 if s > 1e11 else s)
                              e_sec = e / 1e6 if e > 1e14 else (e / 1000.0 if e > 1e11 else e)
                              
                              s_rel = s_sec if s_sec < 86400 else s_sec - t_sub_min
                              e_rel = e_sec if e_sec < 86400 else e_sec - t_sub_min
                              
                              if s_sec > 86400 and t_sub_min < 86400:
                                   min_cust_sec = min(i['start'] for i in subject_intervals)
                                   min_cust_sec = min_cust_sec / 1e6 if min_cust_sec > 1e14 else (min_cust_sec / 1000.0 if min_cust_sec > 1e11 else min_cust_sec)
                                   s_rel = s_sec - min_cust_sec
                                   e_rel = e_sec - min_cust_sec
                              
                              s_rel += offset_s
                              e_rel += offset_s
                              
                              if s_rel > self.df_full['time_rel'].max() + 100: continue
                              
                              if gantt:
                                  if (e_rel - s_rel) < 1.0: continue
                                  if e_rel < t_start_view or s_rel > t_end_view: continue
                                  
                              processed_intervals.append({
                                  'start': s_rel, 'end': e_rel, 'color': color, 'label': lbl, 'duration': e_rel - s_rel
                              })
                              
                         if gantt and processed_intervals:
                              processed_intervals.sort(key=lambda x: x['duration'], reverse=True)
                              valid_y = data_arr[~np.isnan(data_arr)]
                              if len(valid_y) > 0:
                                  y_min = np.nanmin(valid_y)
                                  y_max = np.nanmax(valid_y)
                                  bar_height = (y_max - y_min) * 0.05 if y_max > y_min else 0.8
                                  if bar_height < 0.1: bar_height = 0.5
                                  
                                  # --- FIX: AGRUPACIÓN POR APLICACIÓN (CARRIL ÚNICO Y-AXIS) ---
                                  import re
                                  unique_colors = []
                                  for ival in processed_intervals:
                                      if ival['color'] not in unique_colors:
                                          unique_colors.append(ival['color'])
                                          
                                  drawn_text = set()
                                  
                                  for ival in processed_intervals:
                                      # En lugar de usar 'idx' (baja una fila por cada micro-evento), 
                                      # usamos 'color_idx' para que toda la misma app comparta un único carril horizontal.
                                      color_idx = unique_colors.index(ival['color'])
                                      y_level = y_min - bar_height - color_idx * (bar_height * 1.5)
                                      
                                      # Limpiar sufijos numéricos ej: "OGAMA (1)" -> "OGAMA" para unificar la leyenda
                                      base_label = re.sub(r'\s*\(\d+\)$', '', ival['label'])
                                      
                                      label_str = base_label if base_label not in unique_labels else None
                                      if label_str: unique_labels.add(base_label)
                                      
                                      ax.hlines(y=y_level, xmin=ival['start'], xmax=ival['end'], colors=ival['color'], linewidth=12, label=label_str)
                                      
                                      # Dibujar el texto solo una vez por carril para evitar la mancha borrosa
                                      if base_label not in drawn_text and ival['duration'] > 1.0:
                                          ax.text(x=ival['start'], y=y_level + (bar_height * 0.2), s=base_label, fontsize=7, va='bottom', ha='left', color=ival['color'])
                                          drawn_text.add(base_label)
                                  
                                  # El límite inferior ahora depende solo de la cantidad de apps, protegiendo la gráfica
                                  ax.set_ylim(bottom=y_min - len(unique_colors) * (bar_height * 1.5) - bar_height, top=y_max)
                              
                         elif not gantt:
                              import re
                              for ival in processed_intervals:
                                  base_label = re.sub(r'\s*\(\d+\)$', '', ival['label'])
                                  label_str = base_label if base_label not in unique_labels else None
                                  if label_str: unique_labels.add(base_label)
                                  
                                  ax.axvspan(ival['start'], ival['end'], color=ival['color'], alpha=alpha_span, label=label_str, zorder=1)
                                  ax.axvline(ival['start'], color=ival['color'], linestyle='--', alpha=0.5, zorder=2)
                    except Exception as err:
                         print(f"Error procesando intervalos personalizados: {err}")

                valid_mask = ~np.isnan(data_arr)
                time_vals = time_arr[valid_mask]
                data_vals = data_arr[valid_mask]

                # Outlier Detector (Z-Score > 3 + Límites Biológicos)
                outlier_mask = np.zeros(len(data_vals), dtype=bool)
                if len(data_vals) > 0:
                    try:
                        z_scores = np.abs(sp_stats.zscore(data_vals))
                        outlier_mask = z_scores > 3
                        
                        # Límites Biológicos Estáticos (Imposibles)
                        if col == "heart_rate_bpm":
                            outlier_mask |= (data_vals < 30) | (data_vals > 220)
                        elif col == "conductance_us":
                            outlier_mask |= (data_vals < 0) | (data_vals > 80)
                        elif col == "resistance_ohm":
                            outlier_mask |= (data_vals < 0)

                        outliers_count = np.sum(outlier_mask)
                        if outliers_count > 0 and hasattr(self, 'statusBar') and self.statusBar is not None:
                            self.statusBar.showMessage(f"⚠ Alerta: Se detectaron {outliers_count} outliers en {col}. Usa 'Eliminar Puntos'.", 8000)
                    except Exception: pass
                
                # Usar color personalizado si existe
                if not self.is_dark and col in self.custom_colors:
                    clr = self.custom_colors[col]

                if col == "resistance_ohm": 
                    data_vals = data_vals / 1000.0
                    yl = "kΩ"

                current_df_indices = index_arr[valid_mask]
                n_points = len(time_vals)
                if n_points > 5000:
                    step = max(1, n_points // 5000)
                    t_plot = time_vals[::step]
                    d_plot = data_vals[::step]
                    p_indices = current_df_indices[::step]
                else:
                    t_plot = time_vals
                    d_plot = data_vals
                    p_indices = current_df_indices

                lines = ax.plot(t_plot, d_plot, color=clr, linewidth=1.2, label=title, picker=True)
                
                # Visualización de Outliers (Puntos Rojos)
                step_idx = max(1, n_points // 5000) if n_points > 5000 else 1
                masked_outliers = outlier_mask[::step_idx]
                if np.any(masked_outliers) and getattr(self, 'delete_mode', False):
                    # Dibujar X rojas sobre los puntos outliers
                    ax.scatter(t_plot[masked_outliers], d_plot[masked_outliers], 
                               color=self.colors["red"], s=18, zorder=5, 
                               label="Atípico/Imposible", marker="x")
                if lines:
                    self.picker_data[lines[0]] = {'indices': p_indices, 'col': col, 'sheet': self.active_sheet}

                # --- Guillotina de Outliers Arrastrable ---
                if hasattr(self, 'chk_threshold_line') and self.chk_threshold_line.isChecked() and col == self.cmb_thresh_var.currentText():
                    cur_ylim = ax.get_ylim()
                    init_y = (cur_ylim[0] + cur_ylim[1]) / 2.0 if getattr(self, 'threshold_y', None) is None else self.threshold_y
                    self.threshold_y = init_y
                    self.hline = ax.axhline(init_y, color=self.colors["red"], linestyle='--', linewidth=2, picker=5, label="Umbral")
                    
                    # Conexión Canvas Mouse
                    fig.canvas.mpl_connect('button_press_event', self._on_mouse_event)
                    fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_event)
                    fig.canvas.mpl_connect('button_release_event', self._on_mouse_event)
                ax.set_ylabel(yl, fontsize=9)

                if len(data_vals) > 0:
                    mean_val = np.mean(data_vals)
                    ax.axhline(mean_val, color=clr, linestyle="--", alpha=0.4)
                    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
                    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

                ax.set_xlabel("Tiempo (s)", fontsize=10)

                is_first = (i == 0)
                self._embed_figure("signals", fig, title=f"{title}", force_clear=is_first, color_hex=clr)
                
                with open(self.log_path, "a") as f:
                    f.write(f"Loop completado para {col}\n")
                    f.flush()
            
            except Exception as e:
                import traceback
                print(f"Error al graficar {col}: {e}")
                error_trace = traceback.format_exc()
                print(error_trace)
                if hasattr(self, 'statusBar'):
                    self.statusBar.showMessage(f"Error con {col}: {e}", 5000)
                with open(self.log_path, "a") as f:
                     f.write(f"EXCEPTION IN VARIABLE {col}:\n{error_trace}\n")

        # Guardar conteo final para diagnóstico
        layout_test = self.plot_hosts.get("signals")
        with open(self.log_path, "a") as f:
             if hasattr(layout_test, 'dock_widgets'):
                  f.write(f"Final Signals Docking Widget Count: {len(layout_test.dock_widgets)}\n")
                  f.write(f"Dashboard Area Height: {layout_test.height()}\n")
                  f.flush()

    def _plot_descriptive(self):
        if self.df is None or self.df.empty: return
        cols = self._active_vars()
        if not cols: return
        
        try:
            fig, axes = self._create_fig(rows=max(2, 1 + len(cols)), cols=1, height_per=2.5)
            ax_table = axes[0]; ax_table.axis("off")

            with open(self.log_path, "a") as f:
                f.write(f"--- INICIANDO DESCRIPTIVOS --- cols={cols}\n")
                if self.df is not None:
                    f.write(f"df columns = {self.df.columns.tolist()}\n")
                f.flush()

            stats_data = []
            valid_cols = []

            for col in cols:
                try:
                    if self.signals and not self.force_resample:
                        _, y_arr, *_ = self.signals.get(col, ([], [], []))
                        s = pd.Series(y_arr)
                    else:
                        s = self.df[col].dropna()
                    with open(self.log_path, "a") as f:
                        f.write(f"Variable {col}: shape={s.shape}, empty={s.empty}\n")
                        if not s.empty:
                            f.write(f"Sample values: {s.values[:3]}, dtype={s.dtype}\n")
                        f.flush()
                    stats_data.append([
                        col, f"{len(s)}", f"{s.mean():.3f}", f"{s.std():.3f}", f"{s.min():.3f}", 
                        f"{np.percentile(s, 25):.3f}", f"{s.median():.3f}", f"{np.percentile(s, 75):.3f}", f"{s.max():.3f}"
                    ])
                    valid_cols.append(col)
                except Exception as e:
                    print(f"Error calculando estadística para {col}: {e}")

            with open(self.log_path, "a") as f:
                f.write(f"Stats data size: {len(stats_data)}, Valid cols: {valid_cols}\n")
                f.flush()

            if not stats_data: return

            table = ax_table.table(cellText=stats_data, colLabels=["Variable", "N", "Media", "Desv.Est", "Mín", "Q1", "Mediana", "Q3", "Máx"], 
                                   cellLoc="center", loc="center")
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
            for (row, col_idx), cell in table.get_celld().items():
                cell.set_edgecolor(self.colors["border"])
                if row == 0: 
                    cell.set_facecolor(self.colors["accent"]); cell.set_text_props(color=self.colors["bg"], fontweight="bold")
                else: 
                    cell.set_facecolor(self.colors["surface"]); cell.set_text_props(color=self.colors["text"])
            ax_table.set_title("Estadísticas Descriptivas", fontsize=11, fontweight="bold", pad=10)

            for i, col in enumerate(valid_cols):
                try:
                    ax = axes[1 + i] if (1 + i) < len(axes) else axes[-1]; yl, clr, title = self._var_meta(col)
                    data = self.df[col].dropna().values
                    
                    # Estética Avanzada: Añadir Violín transparente de fondo
                    if len(data) > 3:
                        try:
                            parts = ax.violinplot(data, vert=False, showmeans=False, showmedians=False, showextrema=False)
                            for pc in parts['bodies']:
                                pc.set_facecolor(clr); pc.set_edgecolor(clr); pc.set_alpha(0.15)
                        except: pass

                    ax.boxplot(data, vert=False, patch_artist=True, tick_labels=[title],
                               boxprops=dict(facecolor=clr, alpha=0.45, edgecolor=clr),
                               medianprops=dict(color=self.colors["yellow"], linewidth=2), 
                               whiskerprops=dict(color=clr), capprops=dict(color=clr),
                               flierprops=dict(marker="o", markerfacecolor=self.colors["red"], markersize=3, alpha=0.6))
                    ax.set_title(title, fontsize=9, fontweight="bold", loc="left"); ax.set_xlabel(yl, fontsize=8)
                except Exception as e:
                    print(f"Error graficando caja para {col}: {e}")

            self._embed_figure("descr", fig, title="Estadísticas Descriptivas")
        except Exception as e:
            if hasattr(self, 'statusBar'):
                self.statusBar.showMessage(f"Error en Descriptivos: {e}", 5000)

    def _plot_distribution(self):
        if self.df is None or self.df.empty: return
        cols = self._active_vars()
        if not cols: return
        
        try:
            fig, axes = self._create_fig(rows=len(cols), cols=2, height_per=3.2, width=12)

            for i, col in enumerate(cols):
                try:
                    yl, clr, title = self._var_meta(col)
                    data = self.df[col].dropna().values
                    if len(data) < 5: 
                        axes[i * 2].text(0.5, 0.5, "Datos insuficientes (<5)", ha="center", va="center", color=self.colors["subtext"])
                        axes[i * 2 + 1].text(0.5, 0.5, "Datos insuficientes (<5)", ha="center", va="center", color=self.colors["subtext"])
                        continue

                    ax_hist = axes[i * 2]
                    ax_hist.hist(data, bins=min(50, max(10, len(data) // 20)), color=clr, alpha=0.55, 
                                 edgecolor=self.colors["border"], density=True, label="Datos")
                    
                    mu, sigma = np.mean(data), np.std(data)
                    if sigma > 0:
                        x_fit = np.linspace(data.min(), data.max(), 200)
                        ax_hist.plot(x_fit, sp_stats.norm.pdf(x_fit, mu, sigma), color=self.colors["yellow"], linewidth=2, 
                                     label=f"Normal (μ={mu:.2f})")

                    for p, pc in zip([10, 25, 50, 75, 90], [self.colors["red"], self.colors["peach"], self.colors["green"], 
                                                            self.colors["peach"], self.colors["red"]]):
                        pval = np.percentile(data, p)
                        ax_hist.axvline(pval, color=pc, linestyle="--", alpha=0.8, linewidth=1)
                        ax_hist.text(pval, ax_hist.get_ylim()[1] * 0.92, f"P{p}", fontsize=7, ha="center", color=pc,
                                     bbox=dict(boxstyle="round,pad=0.2", facecolor=self.colors["surface"], alpha=0.8))

                    ax_hist.set_title(f"{title} — Distribución", fontsize=9, fontweight="bold")
                    ax_hist.set_xlabel(yl, fontsize=8); ax_hist.set_ylabel("Densidad", fontsize=8)
                    ax_hist.legend(fontsize=7, facecolor=self.colors["surface"], framealpha=0.8, labelcolor=self.colors["text"])

                    ax_qq = axes[i * 2 + 1]
                    sorted_data = np.sort(data)
                    theoretical_q = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
                    ax_qq.scatter(theoretical_q, sorted_data, c=clr, s=8, alpha=0.6)
                    
                    try:
                        slope, intercept = np.polyfit(theoretical_q, sorted_data, 1)
                        ax_qq.plot(theoretical_q, slope * theoretical_q + intercept, color=self.colors["yellow"], 
                                   linewidth=1.5, linestyle="--", label="Ref. Normal")
                    except: pass

                    ax_qq.set_title(f"{title} — Q-Q Plot", fontsize=9, fontweight="bold")
                    ax_qq.set_xlabel("Cuantiles Teóricos", fontsize=8); ax_qq.set_ylabel("Cuantiles Observados", fontsize=8)
                    ax_qq.legend(fontsize=7, facecolor=self.colors["surface"], framealpha=0.8, labelcolor=self.colors["text"])

                    sample = data[:5000] if len(data) > 5000 else data
                    if len(sample) >= 3:
                        try:
                            w_stat, p_val = sp_stats.shapiro(sample)
                            ax_qq.text(0.03, 0.95, f"Shapiro W={w_stat:.4f}\np={p_val:.4f} {'(OK)' if p_val>0.05 else '(X)'}", 
                                       transform=ax_qq.transAxes, fontsize=7, va="top",
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors["surface2"], alpha=0.9), 
                                       color=self.colors["text"])
                        except: pass
                except Exception as inner_e:
                    print(f"Error en distribución para {col}: {inner_e}")
                    if (i * 2) < len(axes):
                        axes[i * 2].text(0.5, 0.5, f"Error: {str(inner_e)[:30]}", ha="center", va="center", color=self.colors["red"])

            self._embed_figure("dist", fig, title="Análisis de Distribución")
        except Exception as e:
            if hasattr(self, 'statusBar'):
                self.statusBar.showMessage(f"Error en Distribución: {e}", 5000)

    def _align_signals(self, col_x, col_y, tolerance_sec=0.5):
        if self.force_resample or not self.signals:
            mask = self.df[[col_x, col_y]].dropna().index
            if len(mask) == 0: return np.array([]), np.array([]), []
            x = self.df.loc[mask, col_x].values.astype(float)
            y = self.df.loc[mask, col_y].values.astype(float)
            return x, y, mask

        tx, vx, ix = self.signals.get(col_x, (np.array([]), np.array([]), np.array([])))
        ty, vy, iy = self.signals.get(col_y, (np.array([]), np.array([]), np.array([])))
        if len(tx) == 0 or len(ty) == 0: return np.array([]), np.array([]), []

        if len(tx) <= len(ty):
            t_base, v_base, idx_base, t_target, v_target, swapped = tx, vx, ix, ty, vy, False
        else:
            t_base, v_base, idx_base, t_target, v_target, swapped = ty, vy, iy, tx, vx, True

        from scipy.spatial import cKDTree
        tree = cKDTree(t_target.reshape(-1, 1))
        dists, idxs = tree.query(t_base.reshape(-1, 1), distance_upper_bound=tolerance_sec)

        valid_mask = dists <= tolerance_sec
        # Filtrar solo índices válidos (cKDTree retorna len(t_target) si no hay match)
        valid_mask = valid_mask & (idxs < len(t_target))
        
        base_match = v_base[valid_mask]
        target_match = v_target[idxs[valid_mask]]
        indices = idx_base[valid_mask]

        if swapped:
            return target_match, base_match, indices
        else:
            return base_match, target_match, indices

    def _plot_regression(self):
        if self.df is None: return
        self.picker_data = {}  # Limpiar datos anteriores
        x_col, y_col = self.cmb_reg_x.currentText(), self.cmb_reg_y.currentText()
        if not x_col or not y_col: return
        
        x, y, mask = self._align_signals(x_col, y_col)
        
        if len(x) < 3:
            fig, axes = self._create_fig(1, 1)
            axes[0].text(0.5, 0.5, "Datos insuficientes (<3 puntos)", ha="center", va="center", color=self.colors["subtext"], fontsize=12)
            axes[0].axis('off')
            self._embed_figure("reg", fig, title="Aviso")
            return

        fig, axes = self._create_fig(rows=1, cols=2, width=12, height_per=5)
        ax1, ax2 = axes[0], axes[1]

        sc1 = ax1.scatter(x, y, c=self.colors["accent"], s=12, alpha=0.5, edgecolors="none", picker=True)
        if not hasattr(self, 'picker_data'): self.picker_data = {}
        self.picker_data[sc1] = {'indices': mask, 'cols': [x_col, y_col], 'sheet': self.active_sheet}
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x_line, slope * x_line + intercept, color=self.colors["red"], linewidth=2, 
                 label=f"y = {slope:.4f}x + {intercept:.4f}")
        ax1.set_xlabel(x_col, fontsize=10); ax1.set_ylabel(y_col, fontsize=10)
        ax1.set_title("Regresión Lineal", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, facecolor=self.colors["surface"], framealpha=0.8, labelcolor=self.colors["text"])
        ax1.text(0.03, 0.95, f"R² = {r_val**2:.4f}\nr = {r_val:.4f}\np = {p_val:.2e}\nSE = {std_err:.4f}", 
                 transform=ax1.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=self.colors["surface2"], 
                           edgecolor=self.colors["border"], alpha=0.9), color=self.colors["text"])

        predicted = slope * x + intercept
        sc2 = ax2.scatter(predicted, y - predicted, c=self.colors["mauve"], s=10, alpha=0.5, edgecolors="none", picker=True)
        self.picker_data[sc2] = {'indices': mask, 'cols': [x_col, y_col], 'sheet': self.active_sheet}
        ax2.axhline(0, color=self.colors["yellow"], linestyle="--", linewidth=1)
        ax2.set_xlabel("Valores Predichos", fontsize=10); ax2.set_ylabel("Residuales", fontsize=10)
        ax2.set_title("Gráfica de Residuales", fontsize=11, fontweight="bold")

        self._embed_figure("reg", fig, title="Regresión Lineal")

    def _plot_clusters(self):
        if self.df is None: return
        self.picker_data = {}  # Limpiar datos anteriores
        cols = self._active_vars()
        if not cols: return
        fig, axes = self._create_fig(rows=1, cols=2, width=13, height_per=5.5)
        ax_corr, ax_km = axes[0], axes[1]

        if self.force_resample or not self.signals:
            corr_data = self.df[cols].dropna()
            if len(corr_data) < 3: 
                ax_corr.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center", color=self.colors["subtext"])
            else:
                corr_matrix = corr_data.corr()
        else:
            # Calcular matriz de correlación emparejando series irregulares
            corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
            from scipy.stats import pearsonr
            for c1 in cols:
                for c2 in cols:
                    if c1 == c2:
                        corr_matrix.loc[c1, c2] = 1.0
                    else:
                        v1, v2, _ = self._align_signals(c1, c2)
                        if len(v1) > 2:
                            corr, _ = pearsonr(v1, v2)
                            corr_matrix.loc[c1, c2] = corr
                        else:
                            corr_matrix.loc[c1, c2] = np.nan
        
        if 'corr_matrix' in locals() and not corr_matrix.isna().all().all():
            n_vars = len(cols)
            im = ax_corr.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax_corr.set_xticks(range(n_vars)); ax_corr.set_yticks(range(n_vars))
            short_labels = [c[:12] for c in cols]
            ax_corr.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
            ax_corr.set_yticklabels(short_labels, fontsize=7)
            for ii in range(n_vars):
                for jj in range(n_vars):
                    val = corr_matrix.values[ii, jj]
                    txt_color = self.colors["bg"] if abs(val) > 0.5 else self.colors["text"]
                    ax_corr.text(jj, ii, f"{val:.2f}", ha="center", va="center", fontsize=7, color=txt_color)
            cb = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors=self.colors["text"])
            ax_corr.set_title("Matriz de Correlación", fontsize=10, fontweight="bold")

        x_col, y_col = self.cmb_cl_x.currentText(), self.cmb_cl_y.currentText()
        
        try:
             import sklearn
             HAS_SKLEARN = True
        except ImportError:
             HAS_SKLEARN = False

        if not HAS_SKLEARN: 
            ax_km.text(0.5, 0.5, "scikit-learn no instalado", ha="center", va="center", color=self.colors["subtext"])
        elif x_col and y_col:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            x, y, mask = self._align_signals(x_col, y_col)
            
            if len(x) >= 10:
                k = self.spn_k.value()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(np.column_stack([x, y]))
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(X_scaled)
                for cl in range(k):
                    if len(mask) > 0:
                        c_mask = mask[labels == cl]
                    else:
                        c_mask = []
                    sc_km = ax_km.scatter(x[labels == cl], y[labels == cl], 
                                          c=self.colors[PALETTE_KEYS[cl % len(PALETTE_KEYS)]], s=14, alpha=0.6, 
                                          label=f"Cluster {cl + 1}", edgecolors="none", picker=True)
                    self.picker_data[sc_km] = {'indices': c_mask, 'cols': [x_col, y_col], 'sheet': self.active_sheet}
                centers = scaler.inverse_transform(km.cluster_centers_)
                ax_km.scatter(centers[:, 0], centers[:, 1], c=self.colors["yellow"], marker="X", s=120, 
                              edgecolors=self.colors["bg"], linewidths=1.5, label="Centroides", zorder=5)
                ax_km.set_xlabel(x_col, fontsize=9); ax_km.set_ylabel(y_col, fontsize=9)
                ax_km.set_title(f"K-Means (K={k})", fontsize=10, fontweight="bold")
                ax_km.legend(fontsize=7, facecolor=self.colors["surface"], framealpha=0.8, labelcolor=self.colors["text"])
                ax_km.text(0.03, 0.95, f"Inercia: {km.inertia_:.1f}", transform=ax_km.transAxes, fontsize=8, va="top",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors["surface2"], alpha=0.9), 
                           color=self.colors["text"])

        self._embed_figure("clust", fig, title="Análisis de Clusters")

    # ─────────────────────────────────────────────────────────────────────────
    # COMPARACIÓN DE HOJAS MULTI-PARAMÉTRICA
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_comparison(self):
        if not self.dfs: return
        self.picker_data = {}
        if not hasattr(self, 'list_compare_sheets'):
             fig, axes = self._create_fig(1, 1)
             self._embed_figure("compare", fig, title="Controles No Implementados", force_clear=True)
             return
             
        selected_indices = self.list_compare_sheets.selectedIndexes()
        if not selected_indices and len(self.dfs) > 1:
            for i in range(self.list_compare_sheets.count()):
                self.list_compare_sheets.item(i).setSelected(True)
            selected_indices = self.list_compare_sheets.selectedIndexes()
            
        if not selected_indices:
            fig, axes = self._create_fig(1, 1)
            self._embed_figure("compare", fig, title="Aviso")
            return
            
        selected_sheets = [self.list_compare_sheets.item(idx.row()).text() for idx in selected_indices if self.list_compare_sheets.item(idx.row())]
             
        if not hasattr(self, 'cmb_compare_var'): return
        variable = self.cmb_compare_var.currentText()
        if not variable: return

        for sheet in selected_sheets:
            if sheet not in self.dfs: continue
            if variable not in self.dfs[sheet].columns:
                fig, axes = self._create_fig(1, 1)
                self._embed_figure("compare", fig, title="Error")
                return

        comp_type_text = self.cmb_compare_type.currentText() if hasattr(self, 'cmb_compare_type') else "Evolución Temporal"
        t0 = float(self.comp_tmin_var.value()) if hasattr(self, 'comp_tmin_var') else 0.0
        t1 = float(self.comp_tmax_var.value()) if hasattr(self, 'comp_tmax_var') else 9999.0
        if t0 > t1: t0, t1 = t1, t0

        unit, _, title = self._var_meta(variable)
        colors = [self.colors[key] for key in PALETTE_KEYS] * (len(selected_sheets) // len(PALETTE_KEYS) + 1)

        if comp_type_text == "Evolución Temporal":
            fig, axes = self._create_fig(rows=1, cols=1, height_per=6, width=12)
            ax = axes[0]
            ax.set_title(f"Alineación Temporal y Gran Promedio: {title} ({unit})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Tiempo Relativo (s)", fontsize=10)
            ax.set_ylabel(unit, fontsize=10)

            time_grid = np.linspace(t0, t1, 1000)
            interp_list = []

            for idx, sheet in enumerate(selected_sheets):
                if sheet not in self.dfs: continue
                df = self.dfs[sheet]
                valid = df[variable].notna() & (df["time_rel"] >= t0) & (df["time_rel"] <= t1)
                time_arr = df.loc[valid, "time_rel"].values
                data_arr = df.loc[valid, variable].values
                
                if self.multi_normalize_var and len(data_arr) > 0:
                    data_arr = data_arr - np.mean(data_arr)
                    
                line, = ax.plot(time_arr, data_arr, color=colors[idx], linewidth=1.0, alpha=0.25, picker=True)
                self.picker_data[line] = {'indices': df.index[valid], 'col': variable, 'sheet': sheet}
                
                if len(data_arr) > 2:
                    interp_y = np.interp(time_grid, time_arr, data_arr)
                    interp_list.append(interp_y)

            # --- GRAND AVERAGE (Media ± 95% Confidence Interval) ---
            if interp_list:
                stacked = np.vstack(interp_list)
                if stacked.size > 0:
                    n_subjects = len(interp_list)
                    grand_mean = np.nanmean(stacked, axis=0)
                    
                    if n_subjects > 1:
                        grand_std = np.nanstd(stacked, axis=0)
                        # Cálculo de CI 95% usando distribución t de Student
                        t_val = sp_stats.t.ppf(0.975, n_subjects - 1)
                        ci_95 = t_val * (grand_std / np.sqrt(n_subjects))
                    else:
                        ci_95 = 0
                    
                    color_ga = self.colors["text"]
                    ax.plot(time_grid, grand_mean, color=color_ga, linewidth=3.5, 
                            label=f"Consolidado Grupal (Media ± 95% CI, n={n_subjects})", zorder=10)
                    
                    if n_subjects > 1:
                        ax.fill_between(time_grid, grand_mean - ci_95, grand_mean + ci_95, 
                                        color=color_ga, alpha=0.2, zorder=9)

            ax.set_title(f"Figura 1. Evolución Temporal de {title}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Tiempo Relativo (s)", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_xlim(t0, t1)
            
            handles, labels = ax.get_legend_handles_labels()
            # Añadir etiqueta para series individuales en fondo
            if interp_list:
                from matplotlib.lines import Line2D
                handles.append(Line2D([0], [0], color=colors[0], alpha=0.3, linewidth=1.0))
                labels.append(f"Sujetos individuales (n={n_subjects})")

            if handles:
                ax.legend(handles, labels, loc="best", fontsize=9, facecolor=self.colors["surface"], 
                          framealpha=0.9, labelcolor=self.colors["text"])
            self._embed_figure("compare", fig, title="Evolución Temporal")

        elif comp_type_text == "Estadística Descriptiva":
            stats_data = []
            box_data = []
            valid_sheets = []
            pooled_all = []
            
            for idx, sheet in enumerate(selected_sheets):
                if sheet not in self.dfs: continue
                df = self.dfs[sheet]
                valid = df[variable].notna() & (df["time_rel"] >= t0) & (df["time_rel"] <= t1)
                s = df.loc[valid, variable].copy()
                
                if s.empty: continue
                if self.multi_normalize_var:
                    s = s - s.mean()
                
                stats_data.append([sheet, f"{len(s)}", f"{s.mean():.3f}", f"{s.std():.3f}", f"{s.min():.3f}", f"{s.median():.3f}", f"{s.max():.3f}"])
                box_data.append(s.values)
                valid_sheets.append(sheet)
                pooled_all.extend(s.values)

            if not stats_data: return
            
            # --- POOLING GLOBAL ---
            if pooled_all:
                p_arr = np.array(pooled_all)
                stats_data.insert(0, ["Consolidado Grupal", f"{len(p_arr)}", f"{p_arr.mean():.3f}", f"{p_arr.std():.3f}", f"{p_arr.min():.3f}", f"{np.median(p_arr):.3f}", f"{p_arr.max():.3f}"])
                box_data.insert(0, p_arr)
                valid_sheets.insert(0, "Consolidado Grupal")
                colors.insert(0, self.colors["text"])

            import pandas as pd
            df_compare = pd.DataFrame(stats_data, columns=["Serie", "N (t)", "Media", "Desv.Est", "Mín", "Mediana", "Máx"])
            self.last_df_compare = df_compare

            # --- RENDERIZADO DINÁMICO: Altura fija por fila para permitir scroll ---
            num_rows = len(stats_data)
            table_height = max(3.0, (num_rows + 1) * 0.45) # 0.45 inches por fila header incluido
            
            fig_table, axes_table = self._create_fig(rows=1, cols=1, height_per=table_height, width=12)
            ax_table = axes_table[0]
            ax_table.axis("off")

            table = ax_table.table(cellText=stats_data, colLabels=["Serie", "N (t)", "Media", "Desv.Est", "Mín", "Mediana", "Máx"], cellLoc="center", loc="center")
            table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.6)
            for (row, col_idx), cell in table.get_celld().items():
                cell.set_edgecolor(self.colors["border"])
                if row == 0: 
                    cell.set_facecolor(self.colors["accent"]); cell.set_text_props(color=self.colors["bg"], fontweight="bold")
                elif row == 1: # Fila del consolidado
                    cell.set_facecolor(self.colors["surface2"]); cell.set_text_props(color=self.colors["text"], fontweight="bold")
                else: 
                    cell.set_facecolor(self.colors["surface"]); cell.set_text_props(color=self.colors["text"])
            
            ax_table.set_title(f"Métricas Descriptivas Consolidadas (n={num_rows-1})", fontsize=11, fontweight="bold", pad=10)
            self._embed_figure("compare", fig_table, title="Métricas Descriptivas", force_clear=True)

            # --- GRÁFICA DE DISTRIBUCIÓN (Boxplot con altura dinámica) ---
            num_subjects = len(valid_sheets)
            box_height = max(4.0, num_subjects * 0.65) # 0.65 inches por sujeto para evitar aplastamiento
            
            fig_box, axes_box = self._create_fig(rows=1, cols=1, height_per=box_height, width=12)
            ax_box = axes_box[0]

            bplot = ax_box.boxplot(box_data, vert=False, patch_artist=True, tick_labels=valid_sheets, medianprops=dict(color=self.colors["yellow"], linewidth=2), flierprops=dict(marker="o", markerfacecolor=self.colors["red"], markersize=3, alpha=0.5))
            for i, (patch, color) in enumerate(zip(bplot['boxes'], colors)):
                patch.set_facecolor(color)
                patch.set_alpha(0.8 if i == 0 else 0.4) # Resaltar la caja del consolidado
            
            ax_box.set_xlabel(unit, fontsize=10)
            ax_box.set_title(f"Distribución de {title}", fontsize=11, fontweight="bold")
            self._embed_figure("compare", fig_box, title="Visualización de Distribución", force_clear=False)


        elif comp_type_text == "Distribución Densidad":
            fig, axes = self._create_fig(rows=1, cols=1, height_per=6, width=12)
            ax = axes[0]
            ax.set_title(f"Histograma de Densidad Consolidado: {title}", fontsize=12, fontweight="bold")
            ax.set_xlabel(unit, fontsize=10)
            ax.set_ylabel("Densidad probabilística", fontsize=10)

            pooled_all = []
            for idx, sheet in enumerate(selected_sheets):
                if sheet not in self.dfs: continue
                df = self.dfs[sheet]
                valid = df[variable].notna() & (df["time_rel"] >= t0) & (df["time_rel"] <= t1)
                data = df.loc[valid, variable].copy()
                if len(data) < 5: continue
                if self.multi_normalize_var: data = data - data.mean()
                
                data = data.values
                pooled_all.extend(data)
                ax.hist(data, bins=min(50, max(10, len(data) // 20)), color=colors[idx], alpha=0.15, density=True)
                
            if pooled_all:
                p_arr = np.array(pooled_all)
                color_ga = self.colors["text"]
                ax.hist(p_arr, bins=min(100, max(20, len(p_arr) // 20)), color=color_ga, alpha=0.4, density=True, label="Consolidado Grupal")
                try:
                    kde = sp_stats.gaussian_kde(p_arr)
                    x_fit = np.linspace(p_arr.min(), p_arr.max(), 200)
                    ax.plot(x_fit, kde(x_fit), color=color_ga, linewidth=3.5)
                except: pass

            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(loc="best", fontsize=9, facecolor=self.colors["surface"], framealpha=0.9, labelcolor=self.colors["text"])
            self._embed_figure("compare", fig, title="Distribución Densidad")

# ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_group_stats(self):
        if not self.dfs: return
        host = self.plot_hosts.get("group")
        if not host: return
        
        g1_names, g2_names = [], []
        if hasattr(self, 'list_g1'):
            g1_names = [self.list_g1.item(idx.row()).text() for idx in self.list_g1.selectedIndexes() if self.list_g1.item(idx.row())]
        if hasattr(self, 'list_g2'):
            g2_names = [self.list_g2.item(idx.row()).text() for idx in self.list_g2.selectedIndexes() if self.list_g2.item(idx.row())]
                
        variable = self.grp_var_cmb.currentText() if hasattr(self, 'grp_var_cmb') else None
        is_paired = getattr(self.grp_is_paired, 'isChecked', lambda: True)() if hasattr(self, 'grp_is_paired') else True

        if not variable or (is_paired and not g1_names) or (not is_paired and not g1_names and not g2_names):
            fig, axes = self._create_fig(1, 1)
            axes[0].text(0.5, 0.5, "Selecciona los grupos correspondientes y la variable a graficar.\n(Para 'Pareado', selecciona sujetos en Grupo A)", 
                          ha="center", va="center", color=self.colors["red"], fontsize=11, fontweight="bold")
            axes[0].axis('off')
            self._embed_figure("group", fig, title="Aviso")
            return

        metric_name = self.grp_metric_cmb.currentText() if hasattr(self, 'grp_metric_cmb') else "Media"

        def get_metric_value(data):
            if len(data) == 0: return np.nan
            if "Media" in metric_name: return data.mean()
            elif "Mediana" in metric_name: return data.median()
            elif "Máximo" in metric_name: return data.max()
            elif "Mínimo" in metric_name: return data.min()
            elif "Desv" in metric_name: return data.std()
            elif "Varianza" in metric_name: return data.var()
            return np.nan

        if is_paired:
            pre_vals_list, post_vals_list, valid_names = [], [], []
            # Usar los rangos definidos en los SpinBoxes de ESTA pestaña
            t0, t1 = self.grp_tmin_var.value(), self.grp_tmax_var.value()
            t0_post, t1_post = self.grp_tmin_post_var.value(), self.grp_tmax_post_var.value()

            for sheet in g1_names:
                if self.force_resample or not self.project_signals.get(sheet):
                    if sheet not in self.dfs: continue
                    df = self.dfs[sheet]
                    if variable not in df.columns: continue
                    valid_pre = df[variable].notna() & (df["time_rel"] >= t0) & (df["time_rel"] <= t1)
                    data_pre = df.loc[valid_pre, variable].copy()
                    valid_post = df[variable].notna() & (df["time_rel"] >= t0_post) & (df["time_rel"] <= t1_post)
                    data_post = df.loc[valid_post, variable].copy()
                else:
                    sig_dict = self.project_signals.get(sheet, {})
                    if variable not in sig_dict: continue
                    t_arr, y_arr, *_ = sig_dict[variable]
                    data_pre = pd.Series(y_arr[(t_arr >= t0) & (t_arr <= t1)])
                    data_post = pd.Series(y_arr[(t_arr >= t0_post) & (t_arr <= t1_post)])
                
                if len(data_pre) == 0 or len(data_post) == 0: continue
                
                v_pre = get_metric_value(data_pre)
                v_post = get_metric_value(data_post)
                
                if not np.isnan(v_pre) and not np.isnan(v_post):
                    pre_vals_list.append(v_pre)
                    post_vals_list.append(v_post)
                    valid_names.append(sheet)
                    
            pre_vals, post_vals = np.array(pre_vals_list), np.array(post_vals_list)
            vals_g1, vals_g2 = pre_vals, post_vals 
        else:
            def extract_group(names, is_group_b=False):
                vals, pooled, valid_names = [], [], []
                # Para Grupo A usamos Pre-Range, para Grupo B usamos Post-Range (si existe) o el mismo Pre-Range
                if is_group_b:
                    t0, t1 = self.grp_tmin_post_var.value(), self.grp_tmax_post_var.value()
                else:
                    t0, t1 = self.grp_tmin_var.value(), self.grp_tmax_var.value()

                for sheet in names:
                    if self.force_resample or not self.project_signals.get(sheet):
                        if sheet not in self.dfs: continue
                        df = self.dfs[sheet]
                        if variable not in df.columns: continue
                        valid = df[variable].notna() & (df["time_rel"] >= t0) & (df["time_rel"] <= t1)
                        data = df.loc[valid, variable].copy()
                    else:
                        sig_dict = self.project_signals.get(sheet, {})
                        if variable not in sig_dict: continue
                        t_arr, y_arr, *_ = sig_dict[variable]
                        data = pd.Series(y_arr[(t_arr >= t0) & (t_arr <= t1)])

                    if len(data) == 0: continue
                    if self.grp_normalize_var: data = data - data.mean()
                    pooled.extend(data.values)
                    vals.append(get_metric_value(data))
                    valid_names.append(sheet)
                return np.array(vals), np.array(pooled), valid_names

            vals_g1, pool_g1, names_g1 = extract_group(g1_names, is_group_b=False)
            vals_g2, pool_g2, names_g2 = extract_group(g2_names, is_group_b=True)

        unit, _, title = self._var_meta(variable)
        fig, axes = self._create_fig(rows=2, cols=1, height_per=4.5, width=12)
        
        label_a = "Grupo A"
        label_b = "Grupo B"

        if is_paired:
            ax_slope = axes[0]
            ax_slope.set_title(f"A. Análisis Individual de Tendencia: {title}", fontsize=12, fontweight="bold", loc="left")
            ax_slope.set_xticks([1, 2])
            ax_slope.set_xticklabels(["Fase 1 (Pre)", "Fase 2 (Post)"])
            ax_slope.set_ylabel(title, fontsize=11)
            
            # --- SPAGHETTI PLOT (Grayscale friendly) ---
            for i in range(len(pre_vals)):
                color = "#333333" if post_vals[i] > pre_vals[i] else "#999999"
                ls = '-' if post_vals[i] > pre_vals[i] else '--'
                ax_slope.plot([1, 2], [pre_vals[i], post_vals[i]], marker='o', linewidth=0.8, alpha=0.3, color=color, linestyle=ls)
                
            # --- GRAND AVERAGE ---
            if len(pre_vals) > 0:
                m_pre, m_post = np.nanmean(pre_vals), np.nanmean(post_vals)
                sem_pre, sem_post = sp_stats.sem(pre_vals, nan_policy='omit'), sp_stats.sem(post_vals, nan_policy='omit')
                ax_slope.plot([1, 2], [m_pre, m_post], color=self.colors["text"], marker='D', markersize=8, linewidth=4.0, label=f"Total Consolidado (n={len(pre_vals)})", zorder=10)
                ax_slope.fill_between([1, 2], [m_pre - sem_pre, m_post - sem_post], [m_pre + sem_pre, m_post + sem_post], color=self.colors["text"], alpha=0.15)
                ax_slope.legend(loc="best", fontsize=9)
            
            ax_box = axes[1]
            ax_box.set_title(f"B. Distribución Poblacional (Variabilidad)", fontsize=12, fontweight="bold", loc="left")
            plot_data = [pre_vals, post_vals] if len(pre_vals) > 0 else []
            labels_box = ["Fase 1", "Fase 2"]
            colors_box = ["#E0E0E0", "#A0A0A0"]
        else:
            ax_dens = axes[0]
            ax_dens.set_title(f"A. Densidad Probabilística de {title}", fontsize=12, fontweight="bold", loc="left")
            ax_dens.set_xlabel(title, fontsize=11); ax_dens.set_ylabel("Densidad", fontsize=11)
            if len(pool_g1) > 1:
                ax_dens.hist(pool_g1, bins=40, density=True, alpha=0.3, color="#666666", label=label_a, histtype='stepfilled')
            if len(pool_g2) > 1:
                ax_dens.hist(pool_g2, bins=40, density=True, alpha=0.3, color="#000000", label=label_b, histtype='step', linewidth=2)
            ax_dens.legend(loc="best")

            ax_box = axes[1]
            ax_box.set_title(f"B. Comparativa de Grupos Independientes", fontsize=12, fontweight="bold", loc="left")
            plot_data, labels_box, colors_box = [], [], []
            if len(vals_g1) > 0: plot_data.append(vals_g1); labels_box.append(label_a); colors_box.append("#E0E0E0")
            if len(vals_g2) > 0: plot_data.append(vals_g2); labels_box.append(label_b); colors_box.append("#B0B0B0")

        if plot_data and len(plot_data[0]) > 0:
            # --- BOXPLOTS CON TRAMADO (HATCHING) PARA ESCALA DE GRISES ---
            bplot = ax_box.boxplot(plot_data, patch_artist=True, tick_labels=labels_box, medianprops=dict(color="black", linewidth=2.5))
            
            # Tramos diferenciales (Estilo Publicación)
            hatches = ['', '///', '\\\\\\', 'xxx', '...', '---']
            for i, patch in enumerate(bplot['boxes']):
                patch.set_facecolor(colors_box[i % len(colors_box)])
                patch.set_hatch(hatches[i % len(hatches)])
                patch.set_alpha(0.85)

            # --- INYECCIÓN ESTADÍSTICA AUTOMATIZADA ---
            if len(plot_data) == 2:
                try:
                    d1, d2 = plot_data[0], plot_data[1]
                    
                    test_sug, test_reason = self._suggest_statistical_test(d1, d2, is_paired=is_paired)
                    if hasattr(self, 'lbl_stat_suggestion'):
                        self.lbl_stat_suggestion.setText(f"<b>Sugerencia:</b> {test_sug} ({test_reason})")

                    # Llamar al nuevo motor de significancia con rigor científico
                    self._draw_significance_bracket(ax_box, d1, d2, x1=1, x2=2, is_paired=is_paired)
                except Exception as e:
                    print(f"Error en motor estadístico: {e}")

        ax_box.set_ylabel(title, fontsize=11)
        fig.tight_layout()
        self._embed_figure("group", fig, title="Análisis de Serie y Población (Estándar Editorial)")


    def _plot_erd(self):
        source_type = self.cmb_erd_source_type.currentText() if hasattr(self, 'cmb_erd_source_type') else "Log MMST"
        
        has_mmst = hasattr(self, 'mmst_dfs') and self.mmst_dfs
        has_txt = hasattr(self, 'event_txt_dfs') and self.event_txt_dfs
        
        if self.df is None or (source_type == "Log MMST" and not has_mmst) or (source_type == "Eventos TXT" and not has_txt):
            fig, axes = self._create_fig(1, 1)
            self._embed_figure("erd", fig, title=f"Módulo ERD (Requiere Datos y {source_type})")
            return

        pre_window = 2.0
        post_window = 6.0
        n_points = 800
        time_x = np.linspace(-pre_window, post_window, n_points)
        category_data = {} 

        # En ERD, mostramos el sujeto activo o todos si el usuario deseara análisis grupal posterior
        # pero para el selector de arriba, priorizamos el sujeto activo.
        targets = [(self.active_sheet, self.df_full)]
        # Si quisiéramos multi-sujeto aquí, deberíamos añadir una lista como en los otros módulos.
        # Por ahora, restauramos la funcionalidad del selector superior.

        for sheet_name, df_sub in targets:
            if df_sub is None or df_sub.empty: continue
            
            # 1. Vincular Eventos
            mapped_key = self.erd_mappings.get(sheet_name, sheet_name)
            source_dict = self.mmst_dfs if source_type == "Log MMST" else self.event_txt_dfs
            
            events_df = source_dict.get(mapped_key)
            if (events_df is None or events_df.empty) and len(source_dict) == 1:
                events_df = list(source_dict.values())[0] # Fallback único log
            
            if events_df is None or events_df.empty:
                for k, df_ev in source_dict.items():
                    if k.lower() in sheet_name.lower() or sheet_name.lower() in k.lower():
                        events_df = df_ev; break
            if events_df is None or events_df.empty: continue
            
            has_hr = 'heart_rate_bpm' in df_sub.columns
            has_scl = 'conductance_us' in df_sub.columns

            # 2. Sincronización de Tiempos
            if 'timestamp' in df_sub.columns:
                num_ts = pd.to_numeric(df_sub['timestamp'], errors='coerce')
                abs_ts = num_ts[num_ts > 1e8]
                t_sub_min_raw = abs_ts.min() if not abs_ts.empty else num_ts.min()
                if pd.isna(t_sub_min_raw): t_sub_min = df_sub['time_rel'].min()
                elif t_sub_min_raw > 1e14: t_sub_min = t_sub_min_raw / 1e6
                elif t_sub_min_raw > 1e11: t_sub_min = t_sub_min_raw / 1000.0
                else: t_sub_min = t_sub_min_raw
            else:
                t_sub_min = df_sub['time_rel'].min()
                
            time_rel_arr = df_sub['time_rel'].values
            hr_arr = df_sub['heart_rate_bpm'].values if has_hr else None
            scl_arr = df_sub['conductance_us'].values if has_scl else None

            onsets = events_df['timestamp_onset'].values
            cat_col = 'affective_category' if source_type == "Log MMST" else 'event_name'
            cats_raw = events_df[cat_col].values if cat_col in events_df.columns else np.array(['General'] * len(onsets))

            mx_onset = np.nanmax(onsets) if len(onsets) > 0 else 0
            if mx_onset > 1e14: onset_sec_arr = onsets / 1e6
            elif mx_onset > 1e11: onset_sec_arr = onsets / 1000.0
            else: onset_sec_arr = onsets

            # Calculamos onset relativo estándar
            if mx_onset > 86400 and t_sub_min < 86400:
                 onset_rel_arr = onset_sec_arr - np.nanmin(onset_sec_arr)
            else:
                 onset_rel_arr = np.where(onset_sec_arr < 86400, onset_sec_arr, onset_sec_arr - t_sub_min)

            # --- CORRECCIÓN CRÍTICA DE DESFASE ZONA HORARIA ---
            if len(onset_rel_arr) > 0 and mx_onset > 86400:
                valid_mask = (onset_rel_arr >= time_rel_arr.min() - 10) & (onset_rel_arr <= time_rel_arr.max() + 10)
                if not np.any(valid_mask):
                    # El cruce de relojes falló. Anclamos el primer estímulo al inicio de la Fase 2 (Estrés)
                    post_start = self.subject_phases.get(sheet_name, {}).get('post', (200.0, 400.0))[0]
                    onset_rel_arr = (onset_sec_arr - np.nanmin(onset_sec_arr)) + post_start

            offset_s = getattr(self, 'erd_offset_spn', None).value() if hasattr(self, 'erd_offset_spn') else 0.0

            # 3. Extracción de Épocas
            for j in range(len(onsets)):
                onset_rel = onset_rel_arr[j] + offset_s
                raw_cat_str = str(cats_raw[j]).strip().lower()
                
                if 'aversiv' in raw_cat_str or 'negativ' in raw_cat_str: cat = 'Aversiva'
                elif 'neutra' in raw_cat_str: cat = 'Neutra'
                elif 'pacifica' in raw_cat_str or 'positiv' in raw_cat_str: cat = 'Pacíficadora'
                else: cat = 'General'
                    
                if np.isnan(onset_rel): continue
                
                # Ignoramos eventos que caen completamente fuera del registro (con margen de 10s)
                if onset_rel < time_rel_arr.min() - 10 or onset_rel > time_rel_arr.max() + 10: 
                    continue
                
                t0, t1 = onset_rel - pre_window, onset_rel + post_window
                
                idx0 = np.searchsorted(time_rel_arr, t0, side='left')
                idx1 = np.searchsorted(time_rel_arr, t1, side='right')

                if idx1 > idx0 and (idx1 - idx0) > 3:
                    local_time = time_rel_arr[idx0:idx1] - onset_rel
                    
                    if cat not in category_data:
                        category_data[cat] = {'hr': {}, 'scl': {}}
                    if sheet_name not in category_data[cat]['hr']:
                        category_data[cat]['hr'][sheet_name] = []
                        category_data[cat]['scl'][sheet_name] = []

                    if has_hr:
                        hr_frag = hr_arr[idx0:idx1]
                        if not np.isnan(hr_frag).all():
                            mask = ~np.isnan(hr_frag)
                            if np.sum(mask) > 2:
                                hr_interp = np.interp(time_x, local_time[mask], hr_frag[mask])
                                base_mask = time_x < 0
                                if getattr(self, 'chk_erd_normalize', None) and self.chk_erd_normalize.isChecked() and np.any(base_mask):
                                    hr_interp = hr_interp - np.nanmean(hr_interp[base_mask])
                                category_data[cat]['hr'][sheet_name].append(hr_interp)

                    if has_scl:
                        scl_frag = scl_arr[idx0:idx1]
                        if not np.isnan(scl_frag).all():
                            mask = ~np.isnan(scl_frag)
                            if np.sum(mask) > 2:
                                scl_interp = np.interp(time_x, local_time[mask], scl_frag[mask])
                                base_mask = time_x < 0
                                if getattr(self, 'chk_erd_normalize', None) and self.chk_erd_normalize.isChecked() and np.any(base_mask):
                                    scl_interp = scl_interp - np.nanmean(scl_interp[base_mask])
                                category_data[cat]['scl'][sheet_name].append(scl_interp)

        # Validación estricta de diccionarios llenos
        has_data = False
        for cat, sig_dict in category_data.items():
            if any(len(trials) > 0 for trials in sig_dict['hr'].values()) or any(len(trials) > 0 for trials in sig_dict['scl'].values()):
                has_data = True
                break

        if not has_data:
            fig, axes = self._create_fig(1, 1)
            axes[0].text(0.5, 0.5, "El cruce temporal falló.\nVerifica el 'Alineación Offset' o las 'Fases' del sujeto.", ha="center", va="center", color=self.colors["red"], fontweight="bold")
            axes[0].axis('off')
            self._embed_figure("erd", fig, title="Error de Sincronización ERD")
            return

        fig, axes = self._create_fig(rows=1, cols=2, width=14, height_per=6)
        ax_hr, ax_scl = axes[0], axes[1]
        metrics = [(ax_hr, 'hr', '❤ ERD Ritmo Cardíaco (HR)', r'$\Delta$ BPM'),
                   (ax_scl, 'scl', '💧 ERD Conductancia (SCL)', r'$\Delta$ $\mu$S')]

        colors_cat = [self.colors[k] for k in PALETTE_KEYS]

        for ax, key_m, title_m, ylabel_m in metrics:
            ax.set_title(title_m, fontsize=12, fontweight='bold')
            ax.set_xlabel('Tiempo post-estímulo (s)', fontsize=10)
            ax.set_ylabel(ylabel_m, fontsize=10)
            ax.axvline(0, color='red', linestyle='--', linewidth=1.5, zorder=10)
            ax.axhline(0, color=self.colors['subtext'], linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)
            ax.set_xlim([-pre_window, post_window])

            c_idx = 0
            for cat, sig_dict in category_data.items():
                clr = colors_cat[c_idx % len(colors_cat)]
                grand_avg_components = []

                for subj, trials in sig_dict[key_m].items():
                    if not trials: continue
                    subj_stacked = np.vstack(trials)
                    subj_mean = np.nanmean(subj_stacked, axis=0)
                    
                    subj_mean_smooth = uniform_filter1d(subj_mean, size=15)
                    grand_avg_components.append(subj_mean_smooth)
                    
                    ax.plot(time_x, subj_mean_smooth, color=clr, alpha=0.15, linewidth=1.0)

                if grand_avg_components:
                    grand_stacked = np.vstack(grand_avg_components)
                    grand_mean = np.nanmean(grand_stacked, axis=0)
                    grand_sem = sp_stats.sem(grand_stacked, axis=0, nan_policy='omit')
                    
                    ax.plot(time_x, grand_mean, color=clr, linewidth=3.5, label=f"{cat}")
                    ax.fill_between(time_x, grand_mean - grand_sem, grand_mean + grand_sem, color=clr, alpha=0.2)

                c_idx += 1

            ax.grid(True, linestyle='--', alpha=0.2)

        plot_widget = self._embed_figure("erd", fig, title="Event-Related Design (ERD) - Cursos Temporales")
        
        if plot_widget:
            # Limpiar leyendas previas para evitar fuga de memoria
            if hasattr(plot_widget, '_legends'):
                 for leg in plot_widget._legends:
                      leg.deleteLater()
                 plot_widget._legends.clear()
            else:
                 plot_widget._legends = []

            try:
                from PyQt6.QtWidgets import QScrollArea
                scroll = QScrollArea(plot_widget)
                scroll.setWidgetResizable(True)
                scroll.setStyleSheet("QScrollArea { background-color: rgba(30, 30, 40, 180); border: 1px solid rgba(150,150,150,60); border-radius: 6px; }")
                scroll.setGeometry(45, 45, 200, 100)
                
                container = QWidget()
                container.setStyleSheet("background: transparent;")
                lay = QVBoxLayout(container)
                lay.setContentsMargins(6, 6, 6, 6)
                lay.setSpacing(2)
                
                c_idx = 0
                for cat, sig_dict in category_data.items():
                    clr = colors_cat[c_idx % len(colors_cat)]
                    n_subj_hr = len([s for s, t in sig_dict['hr'].items() if t])
                    n_subj_scl = len([s for s, t in sig_dict['scl'].items() if t])
                    n_samples = max(n_subj_hr, n_subj_scl)
                    if n_samples == 0: continue
                    
                    lbl = QLabel(f"● {cat} (N={n_samples} Suj)")
                    lbl.setStyleSheet(f"color: {clr}; font-weight: bold; font-size: 11px; background: transparent;")
                    lay.addWidget(lbl)
                    c_idx += 1
                    
                lay.addStretch()
                scroll.setWidget(container)
                scroll.raise_()
                scroll.show()
                
                if not hasattr(plot_widget, '_legends'): plot_widget._legends = []
                plot_widget._legends.append(scroll)
            except Exception: pass

    def _draw_significance_bracket(self, ax, d1, d2, x1, x2, is_paired=False, force_test=None):
        """Motor estadístico riguroso para boxplots."""
        d1 = pd.Series(d1).dropna().values
        d2 = pd.Series(d2).dropna().values
        
        if len(d1) < 3 or len(d2) < 3: return

        # 1. Identificar si hay forzado manual
        mode = "auto"
        if force_test:
             if "T-Student" in force_test: mode = "student"
             elif "T-Welch" in force_test: mode = "welch"
             elif "Wilcoxon" in force_test: mode = "non-parametric"

        # 2. Selección de Test
        if mode == "auto":
            try:
                _, p_norm1 = sp_stats.shapiro(d1)
                _, p_norm2 = sp_stats.shapiro(d2)
                is_normal = (p_norm1 > 0.05 and p_norm2 > 0.05)
            except: is_normal = False
            
            if is_normal:
                if is_paired and len(d1) == len(d2):
                    stat, p_val = sp_stats.ttest_rel(d1, d2)
                    test_name = "T-Student (Par)"
                else:
                    stat, p_val = sp_stats.ttest_ind(d1, d2, equal_var=False)
                    test_name = "Welch T-test"
            else:
                if is_paired and len(d1) == len(d2):
                    try:
                        if np.all(d1 == d2):
                            p_val = 1.0; test_name = "Wilcoxon (Idénticos)"
                        else:
                            res = sp_stats.wilcoxon(d1, d2); p_val = res.pvalue
                            test_name = "Wilcoxon"
                    except: p_val = 1.0; test_name = "Wilcoxon (Err)"
                else:
                    res = sp_stats.mannwhitneyu(d1, d2, alternative='two-sided')
                    p_val = res.pvalue; test_name = "Mann-Whitney U"
        
        elif mode == "student":
             if is_paired and len(d1) == len(d2):
                 stat, p_val = sp_stats.ttest_rel(d1, d2)
                 test_name = "T-Student (Par)"
             else:
                 stat, p_val = sp_stats.ttest_ind(d1, d2, equal_var=True)
                 test_name = "T-Student (Indep)"
        
        elif mode == "welch":
             # Welch siempre asume varianza desigual, ttest_ind con equal_var=False
             stat, p_val = sp_stats.ttest_ind(d1, d2, equal_var=False)
             test_name = "T-Welch (Forzado)"
        
        elif mode == "non-parametric":
             if is_paired and len(d1) == len(d2):
                 try:
                     if np.all(d1 == d2):
                         p_val = 1.0; test_name = "Wilcoxon (Idénticos)"
                     else:
                         res = sp_stats.wilcoxon(d1, d2); p_val = res.pvalue
                         test_name = "Wilcoxon (Forzado)"
                 except: p_val = 1.0; test_name = "Wilcoxon (F-Err)"
             else:
                 res = sp_stats.mannwhitneyu(d1, d2, alternative='two-sided')
                 p_val = res.pvalue; test_name = "Mann-Whitney (Forzado)"

        # 3. Simbología (p < 0.001 rigor)
        if p_val < 0.001: 
            sig_symbol = "***"
            p_text = "p < 0.001"
        elif p_val < 0.01: 
            sig_symbol = "**"
            p_text = f"p = {p_val:.4f}"
        elif p_val < 0.05: 
            sig_symbol = "*"
            p_text = f"p = {p_val:.4f}"
        else: 
            sig_symbol = "ns"
            p_text = f"p = {p_val:.4f}"

        # Calcular Cohen's d
        if is_paired and len(d1) == len(d2):
            diff = d2 - d1
            std_diff = np.std(diff, ddof=1)
            d_cohen = np.mean(diff) / std_diff if std_diff != 0 else 0
        else:
            n1, n2 = len(d1), len(d2)
            var1, var2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0
            d_cohen = (np.mean(d2) - np.mean(d1)) / pooled_std if pooled_std != 0 else 0
        ef_text = f"d = {abs(d_cohen):.2f}"

        # 4. Dibujo
        y_max = max(np.max(d1), np.max(d2))
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        h = y_range * 0.05 
        y_base = y_max + (y_range * 0.03)
        
        ax.plot([x1, x1, x2, x2], [y_base, y_base+h, y_base+h, y_base], lw=1.5, c=self.colors["text"])
        ax.text((x1+x2)*.5, y_base+h, f"{sig_symbol}\n({p_text}, {ef_text}, {test_name})", 
                ha='center', va='bottom', color=self.colors["text"], fontsize=9, fontweight='bold')
        ax.set_ylim(top=max(ax.get_ylim()[1], y_base + h * 3.5))

    def _plot_phases(self):
        if not self.dfs: return
        variable = self.cmb_phases_var.currentText() if hasattr(self, 'cmb_phases_var') else None
        if not variable: return
        
        test_choice = self.cmb_phases_test.currentText() if hasattr(self, 'cmb_phases_test') else "Automática (Sugerida)"
        alpha_text = self.cmb_phases_alpha.currentText() if hasattr(self, 'cmb_phases_alpha') else "0.05"
        alpha = float(alpha_text.split(" ")[0])
        
        selected_indices = self.list_phases_sheets.selectedIndexes()
        selected_sheets = [self.list_phases_sheets.item(idx.row()).text() for idx in selected_indices if self.list_phases_sheets.item(idx.row())]
        
        if not selected_sheets:
            if self.active_sheet: selected_sheets = [self.active_sheet]
            else: return

        ph1_name = self.cmb_ph1_src.currentText() if hasattr(self, 'cmb_ph1_src') else "Relajación"
        ph2_name = self.cmb_ph2_src.currentText() if hasattr(self, 'cmb_ph2_src') else "MMST"
        
        ph1_label = self.le_ph1_name.text() if (hasattr(self, 'le_ph1_name') and self.le_ph1_name.text()) else "Fase 1"
        ph2_label = self.le_ph2_name.text() if (hasattr(self, 'le_ph2_name') and self.le_ph2_name.text()) else "Fase 2"

        print(f"[DEBUG] _plot_phases called. ph1: {ph1_name}, ph2: {ph2_name}")
        print(f"[DEBUG] subject_phases: {self.subject_phases}")

        # Agregación a Formato Largo (Requisito estricto para LMM y Tests Pareados)
        rows = []
        outliers = getattr(self, 'phases_outliers_removed', set())
        for sheet in selected_sheets:
            if sheet in outliers:
                continue
                
            sheet_phases = self.subject_phases.get(sheet, {})
            p1_range = sheet_phases.get(ph1_name)
            p2_range = sheet_phases.get(ph2_name)
            
            if p1_range is None or p2_range is None:
                continue # Saltar sujeto si no tiene ambas fases seleccionadas
            
            df_sub = self.dfs.get(sheet)
            sig_dict = self.project_signals.get(sheet, {})

            if self.force_resample or not sig_dict or variable not in sig_dict:
                if df_sub is None or df_sub.empty or variable not in df_sub.columns: continue
                s1 = df_sub.loc[(df_sub["time_rel"] >= p1_range[0]) & (df_sub["time_rel"] <= p1_range[1]), variable].dropna()
                s2 = df_sub.loc[(df_sub["time_rel"] >= p2_range[0]) & (df_sub["time_rel"] <= p2_range[1]), variable].dropna()
            else:
                t_arr, y_arr, *_ = sig_dict[variable]
                s1 = pd.Series(y_arr[(t_arr >= p1_range[0]) & (t_arr <= p1_range[1])])
                s2 = pd.Series(y_arr[(t_arr >= p2_range[0]) & (t_arr <= p2_range[1])])
            
            if not s1.empty and not s2.empty:
                rows.append({'Subject': sheet, 'Phase': 'Fase 1', 'Value': s1.mean()})
                rows.append({'Subject': sheet, 'Phase': 'Fase 2', 'Value': s2.mean()})

        if len(rows) < 4:
            fig, axes = self._create_fig(1, 1)
            axes[0].text(0.5, 0.5, "Datos insuficientes. Se requieren al menos 2 sujetos con datos válidos en ambas fases.", 
                         ha="center", va="center", color=self.colors["red"], fontweight="bold")
            axes[0].axis('off')
            self._embed_figure("phases", fig, title="Aviso")
            return

        lmm_df = pd.DataFrame(rows)
        self.last_df_uni = lmm_df
        lmm_df['Phase'] = pd.Categorical(lmm_df['Phase'], categories=['Fase 1', 'Fase 2'], ordered=True)
        yl, clr, title_var = self._var_meta(variable)

        # ---------------------------------------------------------
        # RAMA A: MODELO LINEAL MIXTO (LMM)
        # ---------------------------------------------------------
        if "LMM" in test_choice:
            p_val, fixed_effect_diff = 1.0, 0.0
            summary_text = ""
            lmm_success = False

            try:
                import statsmodels.formula.api as smf
                if lmm_df['Value'].std() < 1e-8:
                    summary_text = "Error LMM: Varianza nula en los datos."
                elif lmm_df['Subject'].nunique() < 2:
                    summary_text = "Aviso LMM: Se requieren >1 sujetos para interceptos aleatorios."
                else:
                    model = smf.mixedlm("Value ~ Phase", lmm_df, groups=lmm_df["Subject"])
                    result = model.fit()
                    p_val = result.pvalues["Phase[T.Fase 2]"]
                    fixed_effect_diff = result.params["Phase[T.Fase 2]"]
                    lmm_success = True
                    summary_text = f"LMM (N={lmm_df['Subject'].nunique()}): Δ={fixed_effect_diff:.3f}, p={p_val:.4f}"
            except Exception as e:
                summary_text = f"Error LMM: {str(e)}"
                print(summary_text)

            fig, _ = self._create_fig(rows=1, cols=1, height_per=6, width=7)
            fig.clf()
            
            show_gantt = getattr(self, 'chk_show_gantt', None) is None or self.chk_show_gantt.isChecked()
            
            if show_gantt:
                gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2.5], hspace=0.3)
                ax_gantt = fig.add_subplot(gs[0])
                ax = fig.add_subplot(gs[1])
                axes = [ax_gantt, ax]
            else:
                gs = fig.add_gridspec(1, 1)
                ax = fig.add_subplot(gs[0])
                axes = [ax]
                
            for ax_t in axes:
                 ax_t.set_facecolor(self.colors["surface"])
                 ax_t.tick_params(colors=self.colors["text"], labelsize=8)
                 ax_t.xaxis.label.set_color(self.colors["text"])
                 ax_t.yaxis.label.set_color(self.colors["text"])
                 ax_t.title.set_color(self.colors["text"])
                 for s_name, spine in ax_t.spines.items():
                      spine.set_color(self.colors["border"])
                 ax_t.grid(True, linestyle='--', color=self.colors["border"], alpha=0.2)

            for subj in lmm_df['Subject'].unique():
                subj_data = lmm_df[lmm_df['Subject'] == subj].sort_values('Phase')
                if len(subj_data) == 2:
                    line, = ax.plot([0, 1], subj_data['Value'].values, marker='o', markersize=4, color=clr, alpha=0.3, linewidth=1.2, zorder=1, picker=5)
                    line.set_gid(subj)

            means = lmm_df.groupby('Phase', observed=True)['Value'].mean()
            ax.plot([0, 1], [means['Fase 1'], means['Fase 2']], marker='D', markersize=8, color=self.colors['text'], linewidth=4.5, zorder=10, label='Efecto Poblacional (Fixed)')

            ax.set_xticks([0, 1]); ax.set_xticklabels([f'{ph1_label}\n(Pre)', f'{ph2_label}\n(Post)'], fontsize=11, fontweight='bold')
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylabel(f"{title_var} ({yl})", fontsize=11)
            title_size = self.spin_lmm_title_size.value() if hasattr(self, 'spin_lmm_title_size') else 13
            pval_offset = self.spin_lmm_pval_offset.value() if hasattr(self, 'spin_lmm_pval_offset') else 0.05

            ax.set_title(f"LMM Longitudinal: {title_var}", fontsize=title_size, fontweight='bold', pad=20)
            
            if lmm_success:
                sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                is_sig = p_val < alpha
                sig_color = self.colors["green"] if is_sig else self.colors["red"]
                
                y_max = lmm_df['Value'].max()
                y_range = y_max - lmm_df['Value'].min()
                ax.text(0.5, y_max + (y_range * pval_offset), f"{sig_str}\n(p={p_val:.4f})", ha='center', va='bottom', fontsize=max(6, title_size-1), fontweight='bold', color=sig_color)
                
                if is_sig:
                    ax.annotate('', xy=(1.05, means['Fase 2']), xytext=(1.05, means['Fase 1']), arrowprops=dict(arrowstyle='<->', color=self.colors["accent"], lw=1.5))
                    ax.text(1.1, (means['Fase 1'] + means['Fase 2'])/2, f"Δ={fixed_effect_diff:+.2f}", va='center', fontweight='bold', color=self.colors["accent"], rotation=270)

            ax.legend(loc='upper left', fontsize=9, framealpha=0.5)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            if show_gantt:
                # --- GANTT CHART DE TIEMPOS RELATIVOS ---
                ax_gantt.set_title("Cronología de Fases por Sujeto", fontsize=11, fontweight='bold', loc='left')
                ax_gantt.set_xlabel("Tiempo Relativo (s)", fontsize=9)
                subjects_list = lmm_df['Subject'].unique()
                y_ticks, y_labels = [], []
                added_labels = set()
                def plot_phase_bar(subj_dict, phase_key, y_offset, height, color, label_text):
                    r = subj_dict.get(phase_key)
                    if r:
                        lbl = label_text if label_text not in added_labels else ""
                        ax_gantt.barh(y_pos + y_offset, width=(r[1] - r[0]), left=r[0], height=height, color=color, alpha=0.7, label=lbl)
                        if lbl: added_labels.add(label_text)

                for i, subj in enumerate(subjects_list):
                    y_pos = len(subjects_list) - i
                    y_ticks.append(y_pos)
                    parts = subj.split('_')
                    y_labels.append(parts[-2] if len(parts) >= 2 else subj[:10])
                    
                    subj_p = self.subject_phases.get(subj, {})
                    plot_phase_bar(subj_p, 'Relajación', 0.1, 0.3, self.colors.get('blue', '#4285F4'), 'Relajación')
                    plot_phase_bar(subj_p, 'MMST', 0.1, 0.3, self.colors.get('orange', '#FF9800'), 'MMST')
                    plot_phase_bar(subj_p, 'OGAMA 1', 0.1, 0.3, self.colors.get('green', '#34A853'), 'OGAMA 1')
                    plot_phase_bar(subj_p, 'OGAMA 2', 0.1, 0.3, self.colors.get('red', '#EA4335'), 'OGAMA 2')
                    plot_phase_bar(subj_p, 'Manual 1', -0.2, 0.2, 'gray', 'Manual 1')
                    plot_phase_bar(subj_p, 'Manual 2', -0.2, 0.2, 'gray', 'Manual 2')

                ax_gantt.set_yticks(y_ticks); ax_gantt.set_yticklabels(y_labels, fontsize=8)
                handles, labels = ax_gantt.get_legend_handles_labels()
                if labels: ax_gantt.legend(loc='upper right', fontsize=8)

            plot_widget = self._embed_figure("phases", fig, title="Reporte Longitudinal LMM")
            if plot_widget and hasattr(plot_widget, 'canvas'):
                plot_widget.canvas.mpl_connect('pick_event', self._on_phase_pick)

        # ---------------------------------------------------------
        # RAMA B: PRUEBAS BIVARIADAS (Clásicas)
        # ---------------------------------------------------------
        else:
            d1_list, d2_list = [], []
            for subj in lmm_df['Subject'].unique():
                subj_data = lmm_df[lmm_df['Subject'] == subj].sort_values('Phase')
                if len(subj_data) == 2:
                    d1_list.append(subj_data.iloc[0]['Value'])
                    d2_list.append(subj_data.iloc[1]['Value'])
            
            d1, d2 = np.array(d1_list), np.array(d2_list)
            
            fig, _ = self._create_fig(rows=1, cols=1, height_per=6, width=12)
            fig.clf()
            
            show_gantt = getattr(self, 'chk_show_gantt', None) is None or self.chk_show_gantt.isChecked()
            
            if show_gantt:
                gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2.5], hspace=0.3)
                ax_gantt = fig.add_subplot(gs[0])
                ax_combined = fig.add_subplot(gs[1])
                axes = [ax_gantt, ax_combined]
            else:
                gs = fig.add_gridspec(1, 1)
                ax_combined = fig.add_subplot(gs[0])
                axes = [ax_combined]
                
            for ax_t in axes:
                 ax_t.set_facecolor(self.colors["surface"])
                 ax_t.tick_params(colors=self.colors["text"], labelsize=8)
                 ax_t.xaxis.label.set_color(self.colors["text"])
                 ax_t.yaxis.label.set_color(self.colors["text"])
                 ax_t.title.set_color(self.colors["text"])
                 for s_name, spine in ax_t.spines.items():
                      spine.set_color(self.colors["border"])
                 ax_t.grid(True, linestyle='--', color=self.colors["border"], alpha=0.2)
            
            # Boxplot Superpuesto con Puntos
            bplot = ax_combined.boxplot([d1, d2], patch_artist=True, widths=0.4, medianprops=dict(color=self.colors["bg"], linewidth=2), zorder=1)
            for patch in bplot['boxes']:
                patch.set_facecolor(clr); patch.set_alpha(0.3)
                
            # Spaghetti Plot sobre el Boxplot
            for subj in lmm_df['Subject'].unique():
                subj_data = lmm_df[lmm_df['Subject'] == subj].sort_values('Phase')
                if len(subj_data) == 2:
                    p1 = subj_data.iloc[0]['Value']
                    p2 = subj_data.iloc[1]['Value']
                    line, = ax_combined.plot([1, 2], [p1, p2], color=clr, alpha=0.5, marker='o', markersize=6, picker=5, zorder=2)
                    line.set_gid(subj)
            
            ax_combined.plot([1, 2], [np.mean(d1), np.mean(d2)], color=self.colors["text"], linewidth=3.5, marker='D', markersize=8, label="Media Grupal", zorder=3)
            ax_combined.set_xticks([1, 2]); ax_combined.set_xticklabels([ph1_label, ph2_label], fontsize=11, fontweight='bold')
            ax_combined.set_title(f"Distribución, Tendencias Individuales y Significancia", fontsize=12, fontweight="bold", pad=15)
            ax_combined.set_ylabel(f"{title_var} ({yl})")
            ax_combined.legend(loc="upper left")
            
            # Sugerencia algorítmica y test forzado
            force_t = None if "Automática" in test_choice else test_choice
            test_sug, reason = self._suggest_statistical_test(d1, d2, is_paired=True)
            
            self._draw_significance_bracket(ax_combined, d1, d2, x1=1, x2=2, is_paired=True, force_test=force_t)
            
            ax_combined.text(0.5, -0.15, f"Sugerencia Algorítmica: {test_sug}\n({reason})", ha='center', va='top', transform=ax_combined.transAxes, fontsize=9, color=self.colors["accent"])
            
            if show_gantt:
                # --- GANTT CHART DE TIEMPOS RELATIVOS ---
                ax_gantt.set_title("Cronología de Fases por Sujeto", fontsize=11, fontweight='bold', loc='left')
                ax_gantt.set_xlabel("Tiempo Relativo (s)", fontsize=9)
                subjects_list = lmm_df['Subject'].unique()
                y_ticks, y_labels = [], []
                added_labels = set()
                def plot_phase_bar(subj_dict, phase_key, y_offset, height, color, label_text):
                    r = subj_dict.get(phase_key)
                    if r:
                        lbl = label_text if label_text not in added_labels else ""
                        ax_gantt.barh(y_pos + y_offset, width=(r[1] - r[0]), left=r[0], height=height, color=color, alpha=0.7, label=lbl)
                        if lbl: added_labels.add(label_text)

                for i, subj in enumerate(subjects_list):
                    y_pos = len(subjects_list) - i
                    y_ticks.append(y_pos)
                    parts = subj.split('_')
                    y_labels.append(parts[-2] if len(parts) >= 2 else subj[:10])
                    
                    subj_p = self.subject_phases.get(subj, {})
                    plot_phase_bar(subj_p, 'Relajación', 0.1, 0.3, self.colors.get('blue', '#4285F4'), 'Relajación')
                    plot_phase_bar(subj_p, 'MMST', 0.1, 0.3, self.colors.get('orange', '#FF9800'), 'MMST')
                    plot_phase_bar(subj_p, 'OGAMA 1', 0.1, 0.3, self.colors.get('green', '#34A853'), 'OGAMA 1')
                    plot_phase_bar(subj_p, 'OGAMA 2', 0.1, 0.3, self.colors.get('red', '#EA4335'), 'OGAMA 2')
                    plot_phase_bar(subj_p, 'Manual 1', -0.2, 0.2, 'gray', 'Manual 1')
                    plot_phase_bar(subj_p, 'Manual 2', -0.2, 0.2, 'gray', 'Manual 2')

                ax_gantt.set_yticks(y_ticks); ax_gantt.set_yticklabels(y_labels, fontsize=8)
                handles, labels = ax_gantt.get_legend_handles_labels()
                if labels: ax_gantt.legend(loc='upper right', fontsize=8)

            plot_widget = self._embed_figure("phases", fig, title="Análisis Bivariado (Pre/Post)")
            if plot_widget and hasattr(plot_widget, 'canvas'):
                plot_widget.canvas.mpl_connect('pick_event', self._on_phase_pick)

    def _plot_multivariate(self):
        if not self.dfs: return
        host = self.plot_hosts.get("multivar")
        if not host: return

        selected_sheets = [item.text() for item in self.list_mv_sheets.selectedItems()]
        selected_vars = [item.text() for item in self.list_mv_vars.selectedItems()]
        
        # FIX: Auto-selección robusta si el usuario no hizo clic manualmente
        if not selected_sheets:
            selected_sheets = list(self.dfs.keys())
            
        if len(selected_vars) < 2:
            if self.list_mv_vars.count() >= 2:
                selected_vars = [self.list_mv_vars.item(0).text(), self.list_mv_vars.item(1).text()]
            else:
                fig, axes = self._create_fig(1, 1)
                axes[0].text(0.5, 0.5, "Requiere al menos 2 variables seleccionadas en el dataset.", ha="center", va="center", color=self.colors["red"], fontweight="bold")
                axes[0].axis('off')
                self._embed_figure("multivar", fig, title="Aviso Multivariado")
                return

        method = self.cmb_mv_method.currentText()
        use_zscore = self.chk_mv_zscore.isChecked()

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from statsmodels.multivariate.manova import MANOVA
        except ImportError:
            fig, axes = self._create_fig(1, 1)
            axes[0].text(0.5, 0.5, "Faltan librerías: instale scikit-learn y statsmodels.", ha="center", va="center", color=self.colors["red"])
            axes[0].axis('off')
            self._embed_figure("multivar", fig, title="Error de Dependencias")
            return

        # ---------------------------------------------------------
        # WRANGLING ESTRICTO
        # ---------------------------------------------------------
        rows = []
        for sheet in selected_sheets:
            df_sub = self.dfs.get(sheet)
            if df_sub is None or df_sub.empty: continue
            
            p_ranges = self.subject_phases.get(sheet, {'Relajación': (0, 100), 'MMST': (100, 250)})
            f1_data, f2_data = {'Subject': sheet, 'Phase': 'Fase 1'}, {'Subject': sheet, 'Phase': 'Fase 2'}
            valid_subject = True
            
            p1 = p_ranges.get('Relajación', (0, 100))
            p2 = p_ranges.get('MMST', (100, 250))
            
            for var in selected_vars:
                if var not in df_sub.columns:
                    valid_subject = False; break
                s1 = df_sub.loc[(df_sub["time_rel"] >= p1[0]) & (df_sub["time_rel"] <= p1[1]), var].dropna()
                s2 = df_sub.loc[(df_sub["time_rel"] >= p2[0]) & (df_sub["time_rel"] <= p2[1]), var].dropna()
                
                if s1.empty or s2.empty:
                    valid_subject = False; break
                f1_data[var] = s1.mean(); f2_data[var] = s2.mean()
                
            if valid_subject:
                rows.append(f1_data); rows.append(f2_data)

        df_mv = pd.DataFrame(rows)
        self.last_df_mv = df_mv
        
        if df_mv.empty or len(df_mv) < 3:
            fig, axes = self._create_fig(1, 1)
            axes[0].text(0.5, 0.5, "Datos insuficientes tras cruzar sujetos, variables y fases.\n(Asegúrate de que los rangos de tiempo de las fases tengan datos).", ha="center", va="center", color=self.colors["red"])
            axes[0].axis('off')
            self._embed_figure("multivar", fig, title="Error de Datos")
            return

        X = df_mv[selected_vars].values
        if use_zscore:
            X = StandardScaler().fit_transform(X)
            df_mv[selected_vars] = X

        # ---------------------------------------------------------
        # A. PCA
        # ---------------------------------------------------------
        if "PCA" in method:
            n_comp = min(len(df_mv), len(selected_vars), 2)
            if n_comp < 2:
                fig, ax = self._create_fig(1, 1)
                ax[0].text(0.5, 0.5, "Se necesitan al menos 3 filas de datos para generar un Biplot 2D.", ha="center", color=self.colors["red"]); ax[0].axis('off')
                self._embed_figure("multivar", fig, title="Aviso PCA")
                return

            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
            df_mv['PC1'], df_mv['PC2'] = components[:, 0], components[:, 1]
            var_exp = pca.explained_variance_ratio_ * 100
            
            fig, ax = self._create_fig(1, 1, width=10, height_per=6)
            ax = ax[0]
            colors_phase = {'Fase 1': self.colors["green"], 'Fase 2': self.colors["red"]}
            for phase in ['Fase 1', 'Fase 2']:
                mask = df_mv['Phase'] == phase
                ax.scatter(df_mv.loc[mask, 'PC1'], df_mv.loc[mask, 'PC2'], c=colors_phase[phase], label=phase, s=80, alpha=0.7, edgecolors='white', zorder=2)

            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            scale_factor = np.max(np.abs(components)) / np.max(np.abs(loadings)) * 0.8
            
            for i, var in enumerate(selected_vars):
                ax.arrow(0, 0, loadings[i, 0] * scale_factor, loadings[i, 1] * scale_factor, color=self.colors["accent"], alpha=0.8, head_width=0.1, zorder=3)
                ax.text(loadings[i, 0] * scale_factor * 1.1, loadings[i, 1] * scale_factor * 1.1, var[:12], color=self.colors["accent"], fontsize=10, fontweight='bold', zorder=4)

            ax.axhline(0, color=self.colors["border"], linestyle='--', lw=1, zorder=1)
            ax.axvline(0, color=self.colors["border"], linestyle='--', lw=1, zorder=1)
            ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% Varianza)", fontsize=11)
            ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% Varianza)", fontsize=11)
            ax.set_title(f"Biplot PCA - Varianza Total Explicada: {sum(var_exp):.1f}%", fontsize=13, fontweight='bold')
            ax.legend(loc='best', facecolor=self.colors["surface"])
            self._embed_figure("multivar", fig, title="PCA")

        # ---------------------------------------------------------
        # B. MANOVA
        # ---------------------------------------------------------
        elif "MANOVA" in method:
            try:
                # Fix Formula OLS (statsmodels no acepta variables empezando con números o símbolos)
                clean_vars = [f"v_{''.join(e for e in v if e.isalnum())}" for v in selected_vars]
                df_manova = df_mv.copy()
                df_manova.columns = ['Subject', 'Phase'] + clean_vars
                formula = f"{' + '.join(clean_vars)} ~ Phase"
                
                manova = MANOVA.from_formula(formula, data=df_manova)
                res = manova.mv_test()
                stats_table = res.results['Phase']['stat'].values
                
                fig, axes = self._create_fig(2, 1, width=11, height_per=4)
                ax_tab, ax_box = axes[0], axes[1]
                ax_tab.axis('off')
                
                cell_text = []
                for i, row_name in enumerate(["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]):
                    cell_text.append([row_name, f"{stats_table[i, 0]:.4f}", f"{stats_table[i, 3]:.4f}", f"{stats_table[i, 4]:.4e}"])
                
                table = ax_tab.table(cellText=cell_text, colLabels=["Métrica Multivariada", "Valor Est.", "F-Value", "P-Value"], loc='center', cellLoc='center')
                table.scale(1, 1.8); table.auto_set_font_size(False); table.set_fontsize(11)
                for (r, c), cell in table.get_celld().items():
                    cell.set_edgecolor(self.colors["border"])
                    if r == 0: cell.set_facecolor(self.colors["accent"]); cell.set_text_props(color=self.colors["bg"], fontweight="bold")
                    else: cell.set_facecolor(self.colors["surface"]); cell.set_text_props(color=self.colors["text"])
                
                ax_tab.set_title("Resultados MANOVA Poblacional", fontweight='bold', fontsize=13)

                import seaborn as sns
                df_melt = pd.melt(df_mv, id_vars=['Subject', 'Phase'], value_vars=selected_vars, var_name='Variable', value_name='Score')
                sns.boxplot(data=df_melt, x='Variable', y='Score', hue='Phase', ax=ax_box, palette=[self.colors["green"], self.colors["red"]])
                ax_box.set_title("Distribución de Efectos Marginales (Estandarizada)", fontweight='bold', fontsize=11)
                ax_box.tick_params(axis='x', rotation=10)
                
                self._embed_figure("multivar", fig, title="MANOVA")
                
            except Exception as e:
                fig, axes = self._create_fig(1, 1)
                axes[0].text(0.5, 0.5, f"Error en MANOVA.\nSe requiere que el N° Sujetos > N° Variables.\n{str(e)}", ha="center", va="center", color=self.colors["red"])
                axes[0].axis('off')
                self._embed_figure("multivar", fig, title="Error MANOVA")

        # ---------------------------------------------------------
        # C. CORRELACIÓN PARCIAL
        # ---------------------------------------------------------
        elif "Correlación" in method:
            corr_matrix = pd.DataFrame(X, columns=selected_vars).corr()
            n_vars = len(selected_vars)
            
            fig, ax = self._create_fig(1, 1, width=8, height_per=6)
            ax = ax[0]
            im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
            
            ax.set_xticks(range(n_vars)); ax.set_yticks(range(n_vars))
            short_labels = [c[:12] for c in selected_vars]
            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(short_labels, fontsize=10)
            
            for ii in range(n_vars):
                for jj in range(n_vars):
                    val = corr_matrix.values[ii, jj]
                    txt_color = self.colors["bg"] if abs(val) > 0.5 else self.colors["text"]
                    ax.text(jj, ii, f"{val:.2f}", ha="center", va="center", fontsize=10, fontweight='bold', color=txt_color)
            
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors=self.colors["text"])
            ax.set_title("Matriz de Correlación Inter-Variables", fontsize=12, fontweight="bold")
            
            self._embed_figure("multivar", fig, title="Matriz de Correlación")


    def _suggest_statistical_test(self, d1, d2, is_paired, is_multivariate=False):
        """
        Clasificador de decisiones estadísticas basado en supuestos de normalidad y homocedasticidad.
        """
        if is_multivariate:
            return "MANOVA", "Evalúa múltiples variables dependientes controlando el Error Tipo I."
            
        try:
            # 1. Supuesto de Normalidad (Shapiro-Wilk)
            _, p_norm1 = sp_stats.shapiro(d1)
            _, p_norm2 = sp_stats.shapiro(d2)
            is_normal = (p_norm1 > 0.05 and p_norm2 > 0.05)
            
            # 2. Supuesto de Homocedasticidad (Levene)
            _, p_levene = sp_stats.levene(d1, d2)
            is_homoscedastic = p_levene > 0.05

            if is_paired:
                if is_normal:
                    return "T-Student Pareada", "Los datos cumplen normalidad intra-sujeto."
                else:
                    return "Wilcoxon", "Violación de normalidad. Se sugiere prueba no paramétrica."
            else:
                if is_normal and is_homoscedastic:
                    return "T-Student Independiente", "Cumple normalidad y varianzas iguales."
                elif is_normal and not is_homoscedastic:
                    return "Welch T-Test", "Cumple normalidad pero viola varianzas iguales (heterocedasticidad)."
                else:
                    return "Mann-Whitney U", "Violación de normalidad en grupos independientes."
        except Exception:
            return "Revisión Manual", "Insuficientes datos para validar supuestos estadísticos."

    # ═══════════════════════════════════════════════════════════════════════════════
    # ERD ANALYSIS METHODS
    # ═══════════════════════════════════════════════════════════════════════════════


    def _extract_signals(self, df):
        sigs = {}
        for col in df.columns:
            if col not in ("time_rel", "timestamp", "sampling_hz"):
                series = df[['time_rel', col]].dropna()
                if not series.empty:
                    sigs[col] = (series['time_rel'].values, series[col].values, series.index.values)
        return sigs

    def load_file(self):
        import re
        file_filter = "Datos (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Seleccionar Archivo(s)", "", file_filter)
        if not paths: return
        
        # Limpiar estructuras anteriores si se carga nuevo conjunto
        errors = []
        loaded_files = []
        if not hasattr(self, 'csv_metadata'):
            self.csv_metadata = {}

        for path in paths:
            try:
                ext = Path(path).suffix.lower()
                stem = Path(path).stem
                if ext in (".xlsx", ".xls"):
                    xls = pd.ExcelFile(path)
                    for sheet in xls.sheet_names:
                        try:
                            df_raw = pd.read_excel(path, sheet_name=sheet)
                            df_clean = self._clean_dataframe(df_raw)
                            if df_clean is not None and not df_clean.empty:
                                key = f"{stem} - {sheet}" if len(xls.sheet_names) > 1 else stem
                                self.dfs[key] = df_clean
                                self.project_signals[key] = self._extract_signals(df_clean)
                                self.file_origins[key] = path
                                loaded_files.append(key)
                        except Exception as e:
                            errors.append(f"Error {stem} ({sheet}): {e}")
                else:
                    try:
                        # Saltar configuración (filas con #) e inyectar metadata
                        skip = 0
                        metadata = {}
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("#"): 
                                    skip += 1
                                    # Extraer clave:valor, e.g. "# Sujeto ID:,1019"
                                    # match de patron con o sin coma despues de los dos puntos
                                    match = re.match(r'^#\s*(.*?):\s*,?\s*(.*)$', line)
                                    if match:
                                        m_key = match.group(1).strip()
                                        m_val = match.group(2).strip()
                                        metadata[m_key] = m_val
                                else: break
                        try:
                            df_raw = pd.read_csv(path, engine='python', sep=None, skipinitialspace=True, skiprows=skip)
                        except Exception:
                            # Fallback si el sniffer de pandas falla en encontrar un delimitador claro
                            df_raw = pd.read_csv(path, skiprows=skip)
                            
                        df_clean = self._clean_dataframe(df_raw)
                        if df_clean is not None and not df_clean.empty:
                            self.dfs[stem] = df_clean
                            self.project_signals[stem] = self._extract_signals(df_clean)
                            self.file_origins[stem] = path
                            self.csv_metadata[stem] = metadata
                            loaded_files.append(stem)
                    except Exception as e:
                        errors.append(f"Error {stem}: {e}")
            except Exception as e:
                errors.append(f"Error procesando {path}: {e}")

        if errors:
            QMessageBox.warning(self, "Errores de Carga", "Algunas series fallaron:\n" + "\n".join(errors[:4]))
        if not self.dfs:
            QMessageBox.critical(self, "Error", "Ningún archivo contiene datos válidos.")
            return

        self.file_path = paths[-1]
        keys = list(self.dfs.keys())
        
        # Bloquear señales para no disparar _set_active_sheet múltiples veces seguidas
        self.cmb_sheet.blockSignals(True)
        self.cmb_sheet.clear()
        self.cmb_sheet.addItems(keys)
        self.cmb_sheet.setCurrentIndex(len(keys) - 1)
        self.cmb_sheet.blockSignals(False)

        self._sync_all_subject_lists(keys)
        self._set_active_sheet(keys[-1])

        # Inicialización de fases basadas en Triggers (RELAXATION y MMST)
        for k in keys:
            df_k = self.dfs[k]
            triggers = df_k.attrs.get('phase_triggers', {})
            
            t_relax_start = triggers.get('RELAXATION_START')
            t_relax_end = triggers.get('RELAXATION_END')
            t_mmst_start = triggers.get('MMST_SESSION_START')
            t_mmst_end = triggers.get('MMST_QUIT')

            valid_pre = None
            if t_relax_start is not None and t_relax_end is not None:
                dur_relax = t_relax_end - t_relax_start
                if dur_relax >= 165.0: # 2m45s mínimo
                    valid_pre = (t_relax_start, t_relax_end)

            valid_post = None
            if t_mmst_start is not None and t_mmst_end is not None:
                dur_mmst = t_mmst_end - t_mmst_start
                if dur_mmst >= 300.0: # 5m mínimo
                    valid_post = (t_mmst_start, t_mmst_end)

            if k not in self.subject_phases:
                self.subject_phases[k] = {}
                
            max_t = df_k['time_rel'].max() if 'time_rel' in df_k.columns else 300.0
                
            if valid_pre is not None:
                self.subject_phases[k]['Relajación'] = valid_pre
            else:
                # Default 3 minutos iniciales para Relajación
                self.subject_phases[k]['Relajación'] = (0.0, min(180.0, max_t))

            if valid_post is not None:
                self.subject_phases[k]['MMST'] = valid_post
            else:
                # Default resto del tiempo para MMST
                self.subject_phases[k]['MMST'] = (min(180.0, max_t), max_t)

        # Mostrar panel de variables de análisis grupal si hay más de 1 individuo
        if hasattr(self, 'group_vars_container'):
             self.group_vars_container.setVisible(len(self.dfs) > 1)

        # DEBUG POPUP PARA EL USUARIO
        debug_msg = "=== RESULTADO DE EXTRACCIÓN DE TRIGGERS ===\n"
        for k in keys:
            df_k = self.dfs[k]
            trig = df_k.attrs.get('phase_triggers', {})
            phases = self.subject_phases.get(k, {})
            debug_msg += f"\nSujeto: {k}\n"
            debug_msg += f"  - Triggers Encontrados: {trig}\n"
            debug_msg += f"  - Fases Válidas: {phases}\n"
        
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Debug Triggers", debug_msg)

        # Mostrar mensaje de éxito en la barra de estado (si existe)
        if hasattr(self, 'statusBar'):
            self.statusBar.showMessage(f"Cargados {len(loaded_files)} archivos: {', '.join(loaded_files)}", 5000)

    def _clean_dataframe(self, df):
        if df is None or df.empty: return None
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        rename_dict = {}
        for col in df.columns:
            for v_info in KNOWN_VARIABLES_META:
                if col == v_info["canonical"] or col in v_info["synonyms"]:
                    rename_dict[col] = v_info["canonical"]
                    break
        if rename_dict: df.rename(columns=rename_dict, inplace=True)
        
        # Corrección de alias para timestamps absolutos
        ts_syns = ["timestamp", "sys_time", "system_time", "unixtime", "unix_time_ms", "time_raw", "absolute_time"]
        for col in df.columns:
            if col.lower().strip() in ts_syns:
                df.rename(columns={col: "timestamp"}, inplace=True)
                break
        
        # --- BÚSQUEDA DE TRIGGERS (FASES) ANTES DE LA CONVERSIÓN ---
        trigger_rows = {}
        for col in df.columns:
            if df[col].dtype == object:
                text_col = df[col].astype(str).str.strip().str.upper()
                for trig in ["RELAXATION_START", "RELAXATION_END", "MMST_SESSION_START", "MMST_QUIT"]:
                    if trig not in trigger_rows:
                        idx_match = text_col[text_col == trig].index
                        if not idx_match.empty:
                            trigger_rows[trig] = idx_match[0]
        # Eliminamos la asignación de df.attrs aquí; se hará después de calcular time_rel

        # Conversión numérica robusta
        for col in df.columns:
            if col not in ("timestamp", "time_rel"):
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- CÁLCULO DE RMSSD Y SDNN DESDE HR (MÉTODO ISÓCRONO) ---
        if "heart_rate_bpm" in df.columns and ("rmssd_ms" not in df.columns or df["rmssd_ms"].isna().all()):
            # Pre-limpiar HR para evitar explosión de varianza en el IBI
            temp_hr = df["heart_rate_bpm"].copy()
            mask_phys_hr = (temp_hr < 30) | (temp_hr > 220)
            m_hr = temp_hr.mean(); s_hr = temp_hr.std()
            mask_z_hr = (temp_hr - m_hr).abs() > (3 * s_hr) if s_hr > 0 else pd.Series([False]*len(df))
            temp_hr.loc[mask_phys_hr | mask_z_hr] = np.nan
            temp_hr = temp_hr.interpolate(method='linear').bfill().ffill()

            fs_hrv = 10.0  
            window_sec = 30  
            window_samples = int(window_sec * fs_hrv)

            # Recuperar el IBI (ms) a partir del BPM
            df['ibi_ms'] = 60000.0 / temp_hr.replace(0, np.nan)

            def calc_rmssd(ibi_array):
                if len(ibi_array) < 2: return np.nan
                diffs = np.diff(ibi_array)
                return np.sqrt(np.mean(diffs**2))

            df['rmssd_ms'] = df['ibi_ms'].rolling(
                window=window_samples, min_periods=window_samples//3
            ).apply(calc_rmssd, raw=True)

            df['sdnn_ms'] = df['ibi_ms'].rolling(
                window=window_samples, min_periods=window_samples//3
            ).std()

            df.drop(columns=['ibi_ms'], inplace=True)
            
            # Transformación logarítmica para análisis paramétrico (LMM, T-Student)
            # Se evita np.log(0) usando replace(0, np.nan)
            if 'rmssd_ms' in df.columns:
                df['ln_rmssd'] = np.log(df['rmssd_ms'].replace(0, np.nan))
            if 'sdnn_ms' in df.columns:
                df['ln_sdnn'] = np.log(df['sdnn_ms'].replace(0, np.nan))

        # --- PRESERVACIÓN DEL DOMINIO TEMPORAL (INTERPOLACIÓN DE OUTLIERS) ---
        outlier_targets = ["heart_rate_bpm", "rmssd_ms", "sdnn_ms"]
        for target in outlier_targets:
            if target in df.columns:
                series = df[target]
                mask_phys = pd.Series([False]*len(df))
                if target == "heart_rate_bpm":
                    mask_phys = (series < 30) | (series > 220)
                else:
                    mask_phys = (series < 0) | (series > 1000)
                
                m = series.mean(); s = series.std()
                mask_z = (series - m).abs() > (3 * s) if s > 0 else pd.Series([False]*len(df))
                
                df.loc[mask_phys | mask_z, target] = np.nan
                # Interpolación Spline Cúbica (Límite 2s -> 120 muestras a 60Hz)
                if df[target].isna().any():
                     if self.force_resample:
                         try:
                             df[target] = df[target].interpolate(method='cubic', limit=120)
                         except:
                             df[target] = df[target].interpolate(method='linear', limit=120)

        # --- FILTRADO AVANZADO EDA/GSR (TÓNICO Y FÁSICO) ---
        if "conductance_us" not in df.columns and "resistance_ohm" in df.columns:
            df["conductance_us"] = 1000.0 / df["resistance_ohm"].replace(0, np.nan)

        if "conductance_us" in df.columns:
            from scipy.signal import butter, filtfilt
            fs = 60.0
            if self.force_resample:
                eda_signal = df["conductance_us"].interpolate().bfill().ffill().values
                valid_idx = df.index
            else:
                eda_series = df["conductance_us"].dropna()
                eda_signal = eda_series.values
                valid_idx = eda_series.index
                
            if len(eda_signal) > 15:
                # 1. Filtro Pasa-Bajos (Tónico) - 1.0 Hz para limpiar ruido
                b_low, a_low = butter(2, 1.0 / (fs / 2.0), btype='low')
                conductance_clean = filtfilt(b_low, a_low, eda_signal)
                df.loc[valid_idx, "conductance_us"] = conductance_clean
                
                # 2. Recálculo SCR (Pasa-Altos) - 0.05 Hz con fase cero (filtfilt)
                b_high, a_high = butter(2, 0.05 / (fs / 2.0), btype='high')
                df.loc[valid_idx, "scr_component"] = filtfilt(b_high, a_high, conductance_clean)
            
            # 3. Cálculo de Z-Scores de Conductancia si no existen
            if "scl_zscore" not in df.columns:
                m_scl = df["conductance_us"].mean()
                s_scl = df["conductance_us"].std()
                if s_scl > 0:
                    df["scl_zscore"] = (df["conductance_us"] - m_scl) / s_scl
                else:
                    df["scl_zscore"] = 0.0
                    
            if "scr_zscore" not in df.columns and "scr_component" in df.columns:
                m_scr = df["scr_component"].mean()
                s_scr = df["scr_component"].std()
                if s_scr > 0:
                    df["scr_zscore"] = (df["scr_component"] - m_scr) / s_scr
                else:
                    df["scr_zscore"] = 0.0
            
        # --- NORMALIZACIÓN TEMPORAL ROBUSTA (ANTIESPAGUETI) ---
        if "time_rel" in df.columns:
            df["time_rel"] = pd.to_numeric(df["time_rel"], errors='coerce')
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')

        if "time_rel" not in df.columns or df["time_rel"].isna().all() or (df["time_rel"].max() > 100000):
            if "timestamp" in df.columns and df["timestamp"].notna().any():
                # 1. Ignorar artefactos: Solo tomar timestamps válidos (UNIX > 100 millones)
                valid_ts_mask = df["timestamp"] > 1e8
                
                if valid_ts_mask.any():
                    # 2. El ancla (ts0) es el mínimo de los valores VÁLIDOS
                    ts0 = df.loc[valid_ts_mask, "timestamp"].min()
                    
                    # 3. Asignar tiempo relativo a las filas válidas
                    # Convertir a segundos si es ms/micro (ts0 > 1e10)
                    scale = 1000.0 if ts0 > 1e11 else 1.0
                    df.loc[valid_ts_mask, "time_rel"] = (df.loc[valid_ts_mask, "timestamp"] - ts0) / scale
                    
                    # 4. Interpolar suavemente los huecos de los artefactos/ceros
                    df["time_rel"] = df["time_rel"].interpolate().bfill().ffill()
                else:
                    df["time_rel"] = np.arange(len(df)) / 60.0
            else:
                df["time_rel"] = np.arange(len(df)) / 60.0
                
        # 5. Anclaje final seguro a 0.0 y ordenamiento cronológico forzado
        df["time_rel"] = df["time_rel"] - df["time_rel"].min()
        
        # Guardar los tiempos exactos de los triggers en attrs antes de reordenar y perder el índice
        final_triggers = {}
        for trig, idx in trigger_rows.items():
            if idx in df.index and pd.notna(df.loc[idx, "time_rel"]):
                final_triggers[trig] = float(df.loc[idx, "time_rel"])
        df.attrs['phase_triggers'] = final_triggers

        df = df.sort_values(by="time_rel").reset_index(drop=True)

        df = df.dropna(subset=["time_rel"])
        return df

    def _set_active_sheet(self, sheet_name):
        if sheet_name not in self.dfs: return
        self.active_sheet = sheet_name
        self.df_full = self.dfs[sheet_name].copy()
        self.df = self.df_full.copy()
        
        # Extraer señales originales del diccionario
        self.signals = self.project_signals.get(sheet_name, {})
        
        if self.signals:
            t_min = min(t[0] for t, *_ in self.signals.values() if len(t) > 0)
            t_max = max(t[-1] for t, *_ in self.signals.values() if len(t) > 0)
        else:
            t_min = float(self.df_full["time_rel"].min())
            t_max = float(self.df_full["time_rel"].max())
            
        if np.isnan(t_min) or np.isinf(t_min): t_min = 0.0
        if np.isnan(t_max) or np.isinf(t_max): t_max = 1.0
        
        # Silenciar señales temporalmente para no plotear doble
        self.spn_start.blockSignals(True)
        self.spn_end.blockSignals(True)
        self.spn_start.setValue(t_min)
        self.spn_end.setValue(t_max)
        self.spn_start.blockSignals(False)
        self.spn_end.blockSignals(False)

        # Asegurar división de fases proporcional a la duración (reset si el archivo cambió mucho)
        p = self.subject_phases.get(sheet_name)
        if not p or p['post'][1] > t_max or p['pre'][1] > t_max:
             mid = (t_min + t_max) / 2.0
             self.subject_phases[sheet_name] = {'pre': (t_min, mid), 'post': (mid, t_max)}

        # Detector de estrés usará el DataFrame remuestreado o las señales crudas (se adaptará detect_stress)
        self.stress_episodes = self.stress_detector.detect_stress(self.df if self.force_resample else self.signals)
        
        if self.signals:
            self.available_cols = list(self.signals.keys())
        else:
            self.available_cols = [c for c in self.df_full.columns if c not in ("time_rel", "timestamp", "sampling_hz") and self.df_full[c].notna().any()]
        self._build_var_controls()
        self._update_combos()
        self._update_phases_events_combo()  # Refrescar ComboBox de fases al cambiar de sujeto
        
        # Actualizar SpinBoxes de Fase para el nuevo activo
        if hasattr(self, 'ph1_start') and sheet_name in self.subject_phases:
             p = self.subject_phases[sheet_name]
             self.ph1_start.blockSignals(True); self.ph1_end.blockSignals(True)
             self.ph2_start.blockSignals(True); self.ph2_end.blockSignals(True)
             
             self.ph1_start.setValue(p['pre'][0]); self.ph1_end.setValue(p['pre'][1])
             self.ph2_start.setValue(p['post'][0]); self.ph2_end.setValue(p['post'][1])
             
             self.ph1_start.blockSignals(False); self.ph1_end.blockSignals(False)
             self.ph2_start.blockSignals(False); self.ph2_end.blockSignals(False)

        fname = Path(self.file_origins.get(sheet_name, self.file_path)).name if self.file_path else "?"
        self.lbl_file.setText(f" 📄 {fname}  |  {len(self.df_full)} filas  |  {t_max:.1f} s")
        self.lbl_file.setStyleSheet(f"font-weight: bold; color: {self.colors['green']};")
        
        # Sincronizar listas de los módulos multi-sujeto
        self._sync_comparative_selections(sheet_name)
        
        # --- Sincronizar Log MMST en módulo ERD para el nuevo sujeto ---
        if hasattr(self, 'cmb_erd_mmst') and hasattr(self, 'erd_mappings'):
            mapped_log = self.erd_mappings.get(sheet_name)
            
            # Si no hay vínculo explícito, intentar autodetectar por similitud de nombre
            if not mapped_log and hasattr(self, 'mmst_dfs') and self.mmst_dfs:
                for k in self.mmst_dfs.keys():
                    if k.lower() in sheet_name.lower() or sheet_name.lower() in k.lower():
                        mapped_log = k
                        break
                        
            # Actualizar la UI del combobox ERD sin disparar eventos en bucle
            if mapped_log:
                self.cmb_erd_mmst.blockSignals(True)
                self.cmb_erd_mmst.setCurrentText(mapped_log)
                self.cmb_erd_mmst.blockSignals(False)
        # ---------------------------------------------------------------

        self._refresh_current_tab()

    def _sync_all_subject_lists(self, keys=None):
        """Puebla y sincroniza todas las QListWidget de sujetos con las claves disponibles."""
        try:
            if keys is None:
                keys = [str(k) for k in self.dfs.keys()] if (hasattr(self, 'dfs') and self.dfs) else []
            else:
                keys = [str(k) for k in keys]
            
            lists_to_sync = ['list_compare_sheets', 'list_phases_sheets', 'list_g1', 'list_g2', 'list_mv_sheets']
            for list_name in lists_to_sync:
                list_w = getattr(self, list_name, None)
                if list_w is not None:
                    # 1. Guardar la selección actual para no perderla
                    selected = [item.text() for item in list_w.selectedItems()]
                    
                    list_w.blockSignals(True)
                    list_w.clear()
                    if keys:
                        list_w.addItems(keys)
                        # 2. Restaurar selecciones
                        restored = False
                        for i in range(list_w.count()):
                            item = list_w.item(i)
                            if item.text() in selected:
                                item.setSelected(True)
                                restored = True
                        
                        # 3. Si no había selección previa, seleccionar el sujeto activo por defecto
                        active = getattr(self, 'active_sheet', None)
                        if not restored and active and active in keys:
                            for i in range(list_w.count()):
                                item = list_w.item(i)
                                if item.text() == active:
                                    item.setSelected(True)
                                    list_w.setCurrentItem(item)
                    list_w.blockSignals(False)
                    list_w.update()
        except Exception as e:
            print(f"[ERROR] Sincronización de listas fallida: {e}")

    def _sync_comparative_selections(self, sheet_name):
        """Mantiene sincronizada la selección en las listas cuando cambia el sujeto activo."""
        lists_to_sync = ['list_compare_sheets', 'list_phases_sheets', 'list_g1', 'list_g2', 'list_mv_sheets']
        for list_name in lists_to_sync:
            list_w = getattr(self, list_name, None)
            if list_w:
                list_w.blockSignals(True)
                found = False
                for i in range(list_w.count()):
                    item = list_w.item(i)
                    is_selected = (item.text() == sheet_name)
                    item.setSelected(is_selected)
                    if is_selected:
                        list_w.setCurrentItem(item)
                        found = True
                list_w.blockSignals(False)

    def _build_var_controls(self):
        # Limpiar Layout de Variables Individuales
        if hasattr(self, 'var_layout'):
             while self.var_layout.count():
                  item = self.var_layout.takeAt(0)
                  if item.widget(): item.widget().deleteLater()
                  
        # Limpiar Layout de Variables Grupales
        if hasattr(self, 'var_group_layout'):
             while self.var_group_layout.count():
                  item = self.var_group_layout.takeAt(0)
                  if item.widget(): item.widget().deleteLater()

        self.var_checks.clear()
        self.var_group_checks.clear()
        found_known = set()
        default_checked = ["heart_rate_bpm", "rmssd_ms", "conductance_us", "scr_component"]
        
        for v_info in KNOWN_VARIABLES_META:
            canonical = v_info["canonical"]
            if canonical in self.available_cols:
                found_known.add(canonical)
                
                # Individual Checkbox
                chk_ind = QCheckBox(v_info["label"])
                chk_ind.setChecked(canonical in default_checked)
                chk_ind.toggled.connect(self._refresh_current_tab)
                self.var_checks[canonical] = chk_ind
                self.var_layout.addWidget(chk_ind)
                
                # Group Checkbox
                chk_grp = QCheckBox(v_info["label"])
                chk_grp.setChecked(canonical in default_checked)
                chk_grp.toggled.connect(self._refresh_current_tab)
                self.var_group_checks[canonical] = chk_grp
                self.var_group_layout.addWidget(chk_grp)
                
        for col in self.available_cols:
            if col not in found_known:
                # Individual Checkbox
                chk_ind = QCheckBox(col)
                chk_ind.setChecked(False)
                chk_ind.toggled.connect(self._refresh_current_tab)
                self.var_checks[col] = chk_ind
                self.var_layout.addWidget(chk_ind)
                
                # Group Checkbox
                chk_grp = QCheckBox(col)
                chk_grp.setChecked(False)
                chk_grp.toggled.connect(self._refresh_current_tab)
                self.var_group_checks[col] = chk_grp
                self.var_group_layout.addWidget(chk_grp)

    def _update_combos(self):
        vals = self.available_cols
        combos = [self.cmb_reg_x, self.cmb_reg_y, self.cmb_cl_x, self.cmb_cl_y]
        if hasattr(self, "cmb_compare_var"): combos.append(self.cmb_compare_var)
        if hasattr(self, "grp_var_cmb"): combos.append(self.grp_var_cmb)
        if hasattr(self, "cmb_phases_var"): combos.append(self.cmb_phases_var)
        for cmb in combos:
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItems(vals)
            cmb.blockSignals(False)

        # Variables para Multivariado (Lista)
        if hasattr(self, "list_mv_vars"):
            self.list_mv_vars.blockSignals(True)
            self.list_mv_vars.clear()
            self.list_mv_vars.addItems(vals)
            self.list_mv_vars.blockSignals(False)
            
        if hasattr(self, "cmb_custom_target"):
            self.cmb_custom_target.blockSignals(True)
            self.cmb_custom_target.clear()
            self.cmb_custom_target.addItems(["Pestañas Generales"] + vals)
            self.cmb_custom_target.blockSignals(False)


    def load_mmst_log(self):
        from PyQt6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(self, "Seleccionar Log(s) MMST/Ogama", "", "CSV (*.csv)")
        if not paths: return
        
        if not hasattr(self, 'mmst_dfs'): self.mmst_dfs = {}
        errors = []
        loaded = []

        for path in paths:
            try:
                skip = 0
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if line.strip().startswith("#"): skip += 1
                        else: break
                df = pd.read_csv(path, skiprows=skip)
                df.columns = df.columns.astype(str).str.strip().str.lower()
                
                synonyms = {
                    'timestamp_onset': ['timestamp_onset', 'onset', 'timeonset', 'start_time', 'time_onset'],
                    'duration_ms': ['duration_ms', 'duration', 'trial_duration', 'length_ms', 'durationms'],
                    'affective_category': ['affective_category', 'category', 'image_type', 'affective_class', 'image_name', 'stimulus']
                }
                
                rename_dict = {}
                for col in df.columns:
                    for target, syn_list in synonyms.items():
                        if col in syn_list: rename_dict[col] = target; break
                if rename_dict: df.rename(columns=rename_dict, inplace=True)

                if 'timestamp_onset' in df.columns:
                    df['timestamp_onset'] = pd.to_numeric(df['timestamp_onset'], errors='coerce')
                if 'duration_ms' in df.columns:
                    df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')

                clean_df = df.dropna(subset=['timestamp_onset']).drop_duplicates(subset=['timestamp_onset'])
                
                from pathlib import Path
                stem = Path(path).stem
                self.mmst_dfs[stem] = clean_df
                loaded.append(stem)

                if self.active_sheet:
                    # Enlace por defecto si sólo hay 1 cargado
                    if len(paths) == 1: self.erd_mappings[self.active_sheet] = stem

            except Exception as e:
                from pathlib import Path
                errors.append(f"{Path(path).name}: {e}")

        # Actualizar Combo en Barra ERD si existe
        if hasattr(self, 'cmb_erd_mmst'):
            self.cmb_erd_mmst.blockSignals(True)
            self.cmb_erd_mmst.clear()
            self.cmb_erd_mmst.addItems(list(self.mmst_dfs.keys()))
            self.cmb_erd_mmst.blockSignals(False)

        if errors: QMessageBox.warning(self, "Atención", "\n".join(errors))
        if loaded: QMessageBox.information(self, "Éxito", f"Logs cargados ({len(loaded)}): " + ", ".join(loaded))
        self._refresh_current_tab()

    def load_siad_log(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from datetime import datetime
        import re
        from pathlib import Path

        paths, _ = QFileDialog.getOpenFileNames(self, "Seleccionar Log SIAD", "", "Logs (*.log *.txt);;All Files (*)")
        if not paths: return

        if not hasattr(self, 'custom_intervals') or isinstance(self.custom_intervals, list):
            self.custom_intervals = {}

        count = 0
        palette_map = {
            "monitor fisiológico": self.colors.get("accent", "#8AB4F8"),
            "gazepointer": self.colors.get("green", "#81C995"),
            "ogama": self.colors.get("peach", "#FCAD70"),
            "psychopy mmst": self.colors.get("red", "#F28B82"),
            "relajación": self.colors.get("teal", "#78D9EC")
        }
        fallback_colors = [self.colors.get(k, '#ffffff') for k in PALETTE_KEYS]

        csv_metadata = getattr(self, 'csv_metadata', {})

        # Construir mapa de búsqueda: df_key -> anchor_unix
        sheet_info = {}
        for k in self.dfs.keys():
            meta = csv_metadata.get(k, {})
            anchor_unix = None
            try:
                unix_val = float(meta.get("UNIX_START", 0))
                if unix_val > 1e8:
                    anchor_unix = unix_val / 1000.0 if unix_val > 1e11 else unix_val
            except (ValueError, TypeError):
                pass
            if anchor_unix is None:
                fecha = meta.get("Fecha Inicio", "").strip()
                if fecha:
                    try:
                        anchor_unix = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S").timestamp()
                    except ValueError:
                        pass
            sheet_info[k] = anchor_unix

        ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
        start_pattern = re.compile(r"Iniciando (.*?) en hilo supervisado", re.IGNORECASE)
        end_pattern = re.compile(r"\[INFO\]\s+SIAD:.*?([A-Za-z][\w\s]+?)\s+ha finalizado", re.IGNORECASE)

        for path in paths:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()

                # FASE 1: Encontrar TODAS las sesiones del log con sus rangos de línea
                sessions = []  # [(start_line_idx, raw_sub, ts_log)]
                for i, line in enumerate(lines):
                    if ("Iniciando grabaci" in line and "UDP para el sujeto" in line) or \
                       ("PROTOCOLO INICIADO" in line and "Sujeto:" in line):
                        ts_match = ts_pattern.search(line)
                        if not ts_match:
                            continue
                        try:
                            ts_log = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
                        except ValueError:
                            continue
                        m = re.search(r"(?:UDP para el sujeto:|Sujeto:)\s*(.+)", line, re.IGNORECASE)
                        if m:
                            sessions.append((i, m.group(1).strip(), ts_log))

                if not sessions:
                    print(f"[SIAD] No se encontraron sesiones en {path}")
                    continue

                # FASE 2: Para cada CSV, encontrar la sesión MÁS CERCANA EN TIEMPO
                for df_key, anchor_unix in sheet_info.items():
                    if anchor_unix is None:
                        print(f"[SIAD] Sin UNIX_START para '{df_key}', omitiendo.")
                        continue

                    # Buscar sesión con menor diferencia temporal al UNIX_START del CSV
                    best_session = min(sessions, key=lambda s: abs(s[2] - anchor_unix))
                    best_sess_idx, best_raw, best_ts = best_session
                    min_diff = abs(best_ts - anchor_unix)

                    # Umbral máximo: 3600s = 1 hora. Si no hay sesión dentro de 1h, ignorar.
                    if min_diff > 3600:
                        print(f"[SIAD] Ninguna sesión dentro de 1h para '{df_key}' (min_diff={min_diff:.0f}s)")
                        continue

                    print(f"[SIAD] CSV '{df_key}' -> sesion '{best_raw}' (diff={min_diff:.0f}s)")

                    # Delimitar rango de esta sesión
                    next_session_line = len(lines)
                    for si, _, _ in sessions:
                        if si > best_sess_idx:
                            next_session_line = si
                            break

                    # Procesar líneas de ESTA sesión
                    current_target_sheet = df_key
                    anchor_ts = anchor_unix  # Usar UNIX_START del CSV como T=0
                    active_apps = {}
                    label_counts = {}

                    if current_target_sheet not in self.custom_intervals:
                        self.custom_intervals[current_target_sheet] = []

                    for line in lines[best_sess_idx:next_session_line]:
                        ts_match = ts_pattern.search(line)
                        if not ts_match:
                            continue
                        try:
                            ts_log = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
                        except ValueError:
                            continue

                        # Inicio de app
                        sm = start_pattern.search(line)
                        if sm:
                            active_apps[sm.group(1).strip()] = ts_log
                            continue

                        # Fin de app
                        if "ha finalizado" in line:
                            em = end_pattern.search(line)
                            if em:
                                app_name = em.group(1).strip()
                                matched_key = None
                                app_clean = app_name.lower()
                                for ak in active_apps:
                                    if ak.lower() == app_clean or app_clean in ak.lower() or ak.lower() in app_clean:
                                        matched_key = ak
                                        break

                                if matched_key:
                                    start_ev_log = active_apps.pop(matched_key)
                                    if app_name.lower() in ["gazepointer", "monitor fisiológico", "compilador mmst"]:
                                        continue

                                    duration = ts_log - start_ev_log
                                    if duration < 5:
                                        continue

                                    adj_start = start_ev_log - anchor_ts
                                    adj_end = ts_log - anchor_ts

                                    if adj_start < -60 or adj_start > 36000:
                                        continue

                                    label_counts[app_name] = label_counts.get(app_name, 0) + 1
                                    lbl = f"{app_name} ({label_counts[app_name]})"
                                    clr = palette_map.get(app_name.lower(), fallback_colors[len(self.custom_intervals[current_target_sheet]) % len(fallback_colors)])

                                    self.custom_intervals[current_target_sheet].append({
                                        'start': adj_start, 'end': adj_end, 'label': lbl, 'color': clr
                                    })
                                    count += 1
                                    print(f"[SIAD] OK: '{lbl}' start={adj_start:.1f}s end={adj_end:.1f}s")

            except Exception as e:
                import traceback
                print(f"Error parseando log SIAD {path}: {e}")
                traceback.print_exc()

        if count > 0:
            if hasattr(self, 'statusBar'): self.statusBar.showMessage(f"Parseo SIAD: {count} eventos vinculados y alineados.", 8000)
            self._update_phases_events_combo()
            for sheet in self.custom_intervals.keys():
                self._auto_assign_phases_from_intervals(sheet)
            self._refresh_current_tab()
        else:
            if hasattr(self, 'statusBar'): self.statusBar.showMessage("No se encontraron eventos o no se pudo emparejar a los sujetos.", 5000)

    def clear_custom_intervals(self):
        if hasattr(self, 'custom_intervals'):
            self.custom_intervals.clear()  # Purgar caché universal, no solo del sujeto activo
        self._update_phases_events_combo()
        self._refresh_current_tab()
        if hasattr(self, 'statusBar'): self.statusBar.showMessage("🧹 Todos los intervalos y logs han sido borrados de la memoria.", 3000)
            
    def _auto_assign_phases_from_intervals(self, sheet_name):
        if not hasattr(self, 'custom_intervals') or not self.custom_intervals: return False
        
        intervals = self.custom_intervals.get(sheet_name, [])
        if not intervals: return False
        
        if sheet_name not in self.subject_phases:
            self.subject_phases[sheet_name] = {}
        
        # Extraer eventos de OGAMA válidos (duración entre 120s y 280s)
        ogama_intervals = []
        for i in intervals:
            if "ogama" in i['label'].lower():
                dur = i['end'] - i['start']
                if 120.0 <= dur <= 280.0:
                    ogama_intervals.append(i)
        
        # Ordenar temporalmente para asignar 1 y 2 correctamente
        ogama_intervals.sort(key=lambda x: x['start'])
        
        if len(ogama_intervals) > 0:
            self.subject_phases[sheet_name]['OGAMA 1'] = (ogama_intervals[0]['start'], ogama_intervals[0]['end'])
        if len(ogama_intervals) > 1:
            self.subject_phases[sheet_name]['OGAMA 2'] = (ogama_intervals[1]['start'], ogama_intervals[1]['end'])
            
        return True

    def _invalidate_subject_variable(self):
        """Marca una variable como inutilizable para el sujeto activo y purga sus datos (convirtiendo a NaN)."""
        if not self.active_sheet: return
        col_to_exclude = self.cmb_thresh_var.currentText()
        
        if self.active_sheet not in self.excluded_signals:
            self.excluded_signals[self.active_sheet] = []
        
        if col_to_exclude not in self.excluded_signals[self.active_sheet]:
            self.excluded_signals[self.active_sheet].append(col_to_exclude)
            
            # Purgar datos en los DataFrames (sustituir por NaN)
            if self.df is not None and col_to_exclude in self.df.columns:
                self.df[col_to_exclude] = np.nan
            if self.df_full is not None and col_to_exclude in self.df_full.columns:
                self.df_full[col_to_exclude] = np.nan
                
                # Regenerar señales base para que las métricas grupales excluyan estos NaNs
                self.project_signals[self.active_sheet] = self._extract_signals(self.df_full)
                self.signals = self.project_signals[self.active_sheet]
                
            if hasattr(self, 'statusBar'):
                self.statusBar.showMessage(f"🚫 Variable {col_to_exclude} purgada para el sujeto {self.active_sheet}.", 5000)
            self._refresh_current_tab()


    def clear_data(self):
        self.dfs.clear()
        self.file_origins.clear()
        self.subject_phases.clear()
        if hasattr(self, 'mmst_dfs'): self.mmst_dfs.clear()
        if hasattr(self, 'custom_intervals'): self.custom_intervals.clear()
        self.df = None
        self.df_full = None
        self.active_sheet = None
        self.cmb_sheet.clear()
        self._sync_all_subject_lists([]) # Limpiar todas las listas de sujetos
        self._refresh_current_tab()

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.colors = DARK_THEME if self.is_dark else LIGHT_THEME
        self._apply_theme()
        # Reflejar el color de fondo en plots existentes
        for tab_name in self.plot_hosts.keys():
             self._refresh_current_tab()

    def export_image(self):
        if hasattr(self, 'fig_cache'):
             path, _ = QFileDialog.getSaveFileName(self, "Exportar Imagen", "grafica.png", "PNG (*.png);;PDF (*.pdf)")
             if path:
                  self.fig_cache.savefig(path, facecolor=self.colors["bg"])

    def _on_sheet_change(self, text):
        if text and text in self.dfs:
            self._set_active_sheet(text)

    def _on_phase_pick(self, event):
        if not hasattr(self, 'chk_clean_phases_outliers') or not self.chk_clean_phases_outliers.isChecked():
            return
        artist = event.artist
        subj = artist.get_gid()
        if subj and hasattr(self, 'phases_outliers_removed'):
            self.phases_outliers_removed.add(subj)
            self._refresh_current_tab()

    def _on_pick(self, event):
        if not getattr(self, 'delete_mode', False):
            return
        artist = event.artist
        if not hasattr(self, 'picker_data') or artist not in self.picker_data:
            return
        p_data = self.picker_data[artist]
        indices = p_data['indices']
        cols = p_data.get('cols', [p_data.get('col')])
        sheet = p_data['sheet']
        if len(event.ind) == 0: return
        pick_idx = event.ind[0]
        if pick_idx >= len(indices): return
        df_idx = indices[pick_idx]

        target_df = self.dfs.get(sheet)
        if target_df is None and sheet == self.active_sheet: target_df = self.df_full
        if target_df is None: return

        prev_values = {}
        for c in cols:
             prev_values[c] = target_df.loc[df_idx, c]
             target_df.loc[df_idx, c] = np.nan
        
        # Sincronizar self.df_full si modificamos active_sheet
        if sheet == self.active_sheet and self.df_full is not None:
             for c in cols:
                  self.df_full.loc[df_idx, c] = np.nan

        # Regenerar las señales originales para ese sheet
        self.project_signals[sheet] = self._extract_signals(target_df)
        if sheet == self.active_sheet:
            self.signals = self.project_signals[sheet]

        self.delete_history.append({
            'sheet': sheet, 'cols': cols, 'df_idx': df_idx, 'prev_values': prev_values
        })
        self.btn_undo.setEnabled(True)
        QTimer.singleShot(10, self._apply_range)

    def _undo_delete(self):
        if not hasattr(self, "delete_history") or not self.delete_history: return
        last = self.delete_history.pop()
        sheet = last.get("sheet")
        
        target_df = self.dfs.get(sheet)
        if target_df is None and sheet == self.active_sheet: target_df = self.df_full
        
        if target_df is not None:
            if "indices" in last and "prev_values" in last: # Formato Bulk/Umbral
                col = last["col"]
                for idx, val in last["prev_values"].items():
                    target_df.loc[idx, col] = val
                    if sheet == self.active_sheet and self.df_full is not None:
                        self.df_full.loc[idx, col] = val
            elif "cols" in last and "df_idx" in last: # Formato Single Point
                idx = last["df_idx"]
                cols = last["cols"]
                vals = last["prev_values"]
                for c in cols:
                    if c in vals:
                        target_df.loc[idx, c] = vals[c]
                        if sheet == self.active_sheet and self.df_full is not None:
                            self.df_full.loc[idx, c] = vals[c]

            # Regenerar señales
            self.project_signals[sheet] = self._extract_signals(target_df)
            if sheet == self.active_sheet:
                self.signals = self.project_signals[sheet]

        if not self.delete_history:
             self.btn_undo.setEnabled(False)
        self._apply_range()

    def _on_mouse_event(self, event):
        if not hasattr(self, 'hline') or self.hline is None: return
        ax = event.inaxes
        if ax is None: return

        if event.name == 'button_press_event':
            if event.button == 1: # Clic izquierdo
                contains, _ = self.hline.contains(event)
                if contains:
                    self.drag_active = True

        elif event.name == 'motion_notify_event':
            if getattr(self, 'drag_active', False) and event.ydata is not None:
                self.threshold_y = event.ydata
                self.hline.set_ydata([event.ydata, event.ydata])
                event.canvas.draw_idle()

        elif event.name == 'button_release_event':
            if getattr(self, 'drag_active', False):
                self.drag_active = False
                col = self.cmb_thresh_var.currentText()
                target_df = self.dfs.get(self.active_sheet) if self.active_sheet in self.dfs else self.df_full
                if target_df is not None and col in target_df.columns:
                    mask_out = target_df[col] > self.threshold_y
                    deleted_indices = target_df.index[mask_out]
                    
                    if len(deleted_indices) > 0:
                        prev_values = {idx: target_df.loc[idx, col] for idx in deleted_indices}
                        self.delete_history.append({
                            'sheet': self.active_sheet, 'col': col, 'indices': deleted_indices, 'prev_values': prev_values
                        })
                        self.btn_undo.setEnabled(True)

                        target_df.loc[mask_out, col] = np.nan
                        if self.df_full is not None and target_df is not self.df_full:
                             self.df_full.loc[mask_out, col] = np.nan
                        
                        if hasattr(self, 'statusBar') and self.statusBar is not None:
                            self.statusBar.showMessage(f"✂ Guillotina: Eliminados {len(deleted_indices)} puntos por encima de {self.threshold_y:.2f}", 4000)

                        QTimer.singleShot(10, self._apply_range)

    def _apply_theme(self):
        if hasattr(self, 'grp_custom_ui'):
            self.grp_custom_ui.setVisible(not self.is_dark)

        self.setStyleSheet(f"""
            /* Fondo General y Texto */
            QMainWindow, QWidget {{ 
                background-color: {self.colors["bg"]}; 
                color: {self.colors["text"]}; 
                font-family: 'Segoe UI', 'Roboto', 'Helvetica', sans-serif;
                font-size: 12px;
            }}
            
            /* Botones Material */
            QPushButton {{
                background-color: {self.colors["surface"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 18px; /* Material Pill */
                padding: 6px 16px;
                color: {self.colors["text"]};
            }}
            QPushButton:hover {{
                background-color: {self.colors["surface2"]};
            }}
            QPushButton:pressed {{
                background-color: {self.colors["border"]};
            }}
            
            /* Toolbar */
            QToolBar {{
                background-color: {self.colors["surface"]};
                border-bottom: 1px solid {self.colors["border"]};
                spacing: 8px;
                padding: 4px;
            }}
            QToolBar QPushButton {{
                border: none;
                border-radius: 4px;
                background-color: transparent;
            }}

            /* Contenedor central de Pestañas (Material Tabs) */
            QTabWidget::pane {{ 
                border: none; 
                background-color: {self.colors["bg"]}; 
            }}
            QTabBar::tab {{ 
                background-color: transparent; 
                color: {self.colors["subtext"]};
                padding: 10px 16px; 
                font-weight: bold;
                border: none;
                border-bottom: 3px solid transparent;
                font-size: {self.font_size_tabs}px;
            }}
            QTabBar::tab:hover {{
                color: {self.colors["text"]};
                background-color: {self.colors["surface2"]};
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{ 
                color: {self.colors["accent"] if not getattr(self, 'tab_selected_color', None) or self.is_dark else self.tab_selected_color}; 
                border-bottom: 3px solid {self.colors["accent"] if not getattr(self, 'tab_selected_color', None) or self.is_dark else self.tab_selected_color};
                background-color: transparent;
            }}

            /* Selectores e Inputs */
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {self.colors["surface"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 4px;
                padding: 5px 10px;
                color: {self.colors["text"]};
            }}
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
                border-color: {self.colors["accent"]};
            }}

            /* QDockWidget (Sidebar) */
            QDockWidget {{
                border: none;
            }}
            QDockWidget::title {{
                background-color: {self.colors["surface"]};
                padding: 8px;
                color: {self.colors["text"]};
                font-weight: bold;
                border-bottom: 1px solid {self.colors["border"]};
            }}

            /* CollapsibleBox Overrides específicos */
            CollapsibleBox QPushButton {{
                background-color: transparent;
                color: {self.colors["accent"]};
                border: none;
                border-radius: 0px;
                text-align: left;
                padding-left: 5px;
            }}
            CollapsibleBox QPushButton:hover {{
                background-color: {self.colors["surface2"]};
            }}
            
            QGroupBox {{ 
                border-radius: 8px; 
                font-weight: bold; 
                margin-top: 15px; 
                color: {self.colors["accent"]}; 
            }}
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # SISTEMA DE PROYECTOS Y PERSISTENCIA (.physio)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _prompt_initial_session(self):
        """Diálogo de inicio para definir la sesión de trabajo."""
        session_name, ok = QInputDialog.getText(self, "Inicio de Sesión", 
                                              "Nombre de la Sesión / Proyecto:", 
                                              QLineEdit.EchoMode.Normal, "")
        if ok and session_name:
            # Buscar si el archivo ya existe en la carpeta actual
            potential_file = f"{session_name}.physio"
            if os.path.exists(potential_file):
                resp = QMessageBox.question(self, "Proyecto Existente", 
                                          f"Se encontró el proyecto '{potential_file}'. ¿Deseas cargarlo?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if resp == QMessageBox.StandardButton.Yes:
                    self.load_project(potential_file)
                    return
            self.setWindowTitle(f"{self.windowTitle()} - [{session_name}]")
            if hasattr(self, 'statusBar'): self.statusBar.showMessage(f"Nueva sesión iniciada: {session_name}")

    def save_project(self, target_path=None):
        """Guarda el estado completo del análisis en un archivo .physio (Pickle)."""
        if not target_path or not isinstance(target_path, str):
            target_path, _ = QFileDialog.getSaveFileName(self, "Guardar Proyecto", "", "Andrade Physio Project (*.physio)")
        
        if not target_path: return
        
        if not target_path.endswith(".physio"): target_path += ".physio"
        
        # Datos a persistir
        state = {
            'dfs': self.dfs,
            'subject_phases': self.subject_phases,
            'erd_mappings': self.erd_mappings,
            'mmst_dfs': getattr(self, 'mmst_dfs', {}),
            'custom_intervals': getattr(self, 'custom_intervals', {}),
            'gantt_mode': getattr(self, 'gantt_mode', True),
            'hide_physio_monitor': getattr(self, 'hide_physio_monitor', False),
            'active_sheet': self.active_sheet,
            'file_path': self.file_path,
            'project_name': os.path.basename(target_path)
        }
        
        try:
            with open(target_path, 'wb') as f:
                pickle.dump(state, f)
            
            self.project_path = target_path
            self._update_recent_files(target_path)
            self._save_session_cache(target_path) 
            
            if hasattr(self, 'statusBar'):
                self.statusBar.showMessage(f"✅ Proyecto guardado: {os.path.basename(target_path)}", 5000)
            self.setWindowTitle(f"Andrade Physio - [{os.path.basename(target_path)}]")
        except Exception as e:
            QMessageBox.critical(self, "Error al Guardar", f"No se pudo guardar el proyecto:\n{e}")

    def load_project(self, target_path=None):
        """Carga y reconstruye un proyecto .physio."""
        if not target_path or not isinstance(target_path, str):
            target_path, _ = QFileDialog.getOpenFileName(self, "Abrir Proyecto", "", "Andrade Physio Project (*.physio)")
        
        if not target_path or not os.path.exists(target_path): return

        try:
            with open(target_path, 'rb') as f:
                state = pickle.load(f)
            
            # Restaurar Atributos
            self.dfs = state.get('dfs', {})
            self.project_signals = state.get('project_signals', {})
            # Retrocompatibilidad: Generar project_signals si el proyecto es de versión anterior
            if not self.project_signals and self.dfs:
                for k, df_old in self.dfs.items():
                    self.project_signals[k] = self._extract_signals(df_old)
                    
            self.subject_phases = state.get('subject_phases', {})
            # Retrocompatibilidad para nombres de fases
            for k, phases in self.subject_phases.items():
                if 'pre' in phases and 'Relajación' not in phases:
                    phases['Relajación'] = phases['pre']
                if 'post' in phases and 'MMST' not in phases:
                    phases['MMST'] = phases['post']
            self.erd_mappings = state.get('erd_mappings', {})
            self.mmst_dfs = state.get('mmst_dfs', {})
            self.custom_intervals = state.get('custom_intervals', {})
            if isinstance(self.custom_intervals, list):
                # Retrocompatibilidad
                self.custom_intervals = {}
            else:
                for sheet in self.custom_intervals.keys():
                    self._auto_assign_phases_from_intervals(sheet)
            self.gantt_mode = state.get('gantt_mode', True)
            self.hide_physio_monitor = state.get('hide_physio_monitor', False)
            self.active_sheet = state.get('active_sheet')
            self.file_path = state.get('file_path')
            
            # Sincronizar UI
            self._sync_ui_after_load()
            
            self.project_path = target_path
            self._update_recent_files(target_path)
            self._save_session_cache(target_path)
            
            self.setWindowTitle(f"Andrade Physio - [{os.path.basename(target_path)}]")
            if hasattr(self, 'statusBar'):
                self.statusBar.showMessage(f"📂 Proyecto cargado: {os.path.basename(target_path)}", 5000)
                
            self._refresh_current_tab()
        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar", f"El archivo .physio está corrupto o es incompatible:\n{e}")

    def _on_phase_event_selected(self, phase_key, event_name):
        if event_name == "Manual (T min/max)":
            self._refresh_current_tab()
            return
            
        selected_sheets = [self.list_phases_sheets.item(idx.row()).text() for idx in self.list_phases_sheets.selectedIndexes()]
        if not selected_sheets: selected_sheets = [self.active_sheet] if self.active_sheet else []
        if not selected_sheets: return

        applied_count = 0
        for sheet in selected_sheets:
            for interval in self.custom_intervals.get(sheet, []):
                if interval['label'] == event_name:
                    if sheet not in self.subject_phases: self.subject_phases[sheet] = {'pre': (0, 100), 'post': (100, 250)}
                    self.subject_phases[sheet][phase_key] = (interval['start'], interval['end'])
                    applied_count += 1
                    
                    if sheet == selected_sheets[-1]:
                        if phase_key == 'pre':
                            self.ph1_start.blockSignals(True); self.ph1_end.blockSignals(True)
                            self.ph1_start.setValue(interval['start']); self.ph1_end.setValue(interval['end'])
                            self.ph1_start.blockSignals(False); self.ph1_end.blockSignals(False)
                        else:
                            self.ph2_start.blockSignals(True); self.ph2_end.blockSignals(True)
                            self.ph2_start.setValue(interval['start']); self.ph2_end.setValue(interval['end'])
                            self.ph2_start.blockSignals(False); self.ph2_end.blockSignals(False)
                    break
                    
        if applied_count > 0 and hasattr(self, 'statusBar'):
            self.statusBar.showMessage(f"✅ '{event_name}' asignado a {applied_count} sujetos (Fase {'1' if phase_key == 'pre' else '2'}).", 5000)
        self._refresh_current_tab()

    def _update_phases_events_combo(self):
        if not hasattr(self, 'cmb_ph1_event') or not hasattr(self, 'cmb_ph2_event'): return
        events = set()
        for intervals in getattr(self, 'custom_intervals', {}).values():
            for interval in intervals: events.add(interval['label'])
                
        ev1, ev2 = self.cmb_ph1_event.currentText(), self.cmb_ph2_event.currentText()
        self.cmb_ph1_event.blockSignals(True); self.cmb_ph2_event.blockSignals(True)
        self.cmb_ph1_event.clear(); self.cmb_ph2_event.clear()
        self.cmb_ph1_event.addItem("Manual (T min/max)")
        self.cmb_ph2_event.addItem("Manual (T min/max)")
        
        for ev in sorted(list(events)):
            self.cmb_ph1_event.addItem(ev); self.cmb_ph2_event.addItem(ev)
            
        if ev1 in events: self.cmb_ph1_event.setCurrentText(ev1)
        if ev2 in events: self.cmb_ph2_event.setCurrentText(ev2)
        self.cmb_ph1_event.blockSignals(False); self.cmb_ph2_event.blockSignals(False)

    def _sync_ui_after_load(self):
        """Reconstruye los combos y listas de sujetos a partir de los datos cargados."""
        if not self.dfs: return
        keys = list(self.dfs.keys())
        self._update_phases_events_combo()
        
        if hasattr(self, 'cmb_sheet'):
            self.cmb_sheet.blockSignals(True)
            self.cmb_sheet.clear()
            self.cmb_sheet.addItems(keys)
            if self.active_sheet in keys:
                 self.cmb_sheet.setCurrentText(self.active_sheet)
            self.cmb_sheet.blockSignals(False)
        
        # Sincronizar listas de sujetos en todos los módulos
        self._sync_all_subject_lists(keys)
            
        # Sincronizar MMST combo en ERD
        if hasattr(self, 'mmst_dfs') and self.mmst_dfs:
             if hasattr(self, 'cmb_erd_mmst'):
                  self.cmb_erd_mmst.blockSignals(True)
                  self.cmb_erd_mmst.clear()
                  self.cmb_erd_mmst.addItems(list(self.mmst_dfs.keys()))
                  self.cmb_erd_mmst.blockSignals(False)

        # Cargar variables en el panel lateral
        current = self.active_sheet if self.active_sheet in keys else keys[0]
        self._set_active_sheet(current)

    def _update_recent_files(self, file_path):
        """Gestiona la lista de archivos recientes en QSettings."""
        recents = self.settings.value("recent_projects", [])
        if not isinstance(recents, list): recents = []
        
        if file_path in recents: recents.remove(file_path)
        recents.insert(0, file_path)
        recents = recents[:10] # Máximo 10
        self.settings.setValue("recent_projects", recents)
        self._rebuild_file_menu()

    def _rebuild_file_menu(self):
        """Actualiza dinámicamente el menú de archivos recientes."""
        if not hasattr(self, 'menu_recent'): return
        self.menu_recent.clear()
        recents = self.settings.value("recent_projects", [])
        if not recents:
            self.menu_recent.addAction("Sin archivos recientes").setEnabled(False)
            return
            
        for path in recents:
            filename = os.path.basename(path)
            action = QAction(filename, self)
            action.triggered.connect(lambda checked, p=path: self.load_project(p))
            self.menu_recent.addAction(action)

    def _save_session_cache(self, path):
        """Guarda la ruta de la última sesión en un archivo oculto."""
        try:
            cache_path = os.path.join(os.getcwd(), ".cache_last_session")
            with open(cache_path, "w") as f:
                f.write(path)
        except: pass

    def _check_session_recovery(self):
        """Verifica si hubo un cierre inesperado y ofrece restaurar."""
        cache_path = os.path.join(os.getcwd(), ".cache_last_session")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    last_path = f.read().strip()
                if os.path.exists(last_path):
                    resp = QMessageBox.question(self, "Recuperación de Sesión", 
                                              "El programa se cerró inesperadamente. ¿Deseas restaurar el último proyecto abierto?",
                                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if resp == QMessageBox.StandardButton.Yes:
                        QTimer.singleShot(1000, lambda: self.load_project(last_path))
            except: pass

if __name__ == "__main__":
    # Fix para que el icono aparezca correctamente en la barra de tareas de Windows
    if sys.platform == 'win32':
        myappid = 'Andrade.PhysioAnalyzer.Pro.V1'
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except: pass

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = PhysioAnalyzerPro()
    window.show()
    sys.exit(app.exec())
