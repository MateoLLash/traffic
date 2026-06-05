"""
app.py - Aplicación Streamlit para análisis de tráfico vehicular y peatonal

Interfaz principal. Todo el CSS/diseño vive en theme.py y components.py.
Este archivo contiene SOLO lógica de negocio y estructura de UI.
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image

# Imports locales — lógica
from detector import ObjectDetector, create_detector
from tracker import ObjectTracker
from counter import TrafficCounter, CountingLine
from exporter import DataExporter
from visualizer import TrafficVisualizer
from utils import (setup_logger, create_output_dirs, get_video_properties,
                  validate_video_file, estimate_processing_time, OBJECT_CLASSES)

# ── DISEÑO ─────────────────────────────────────────────────────────────────────
from theme import (apply_theme, header, section_title, sidebar_label,
                   status_badge, metric_card, video_info_card,
                   step_indicator, live_metrics_panel)
from components import video_player, sidebar_brand, empty_state
from chunker import (VideoChunker, consolidate_stats, consolidate_crossings,
                     get_video_info, needs_chunking, ffmpeg_available,
                     format_duration)

# Configurar página — siempre primero
st.set_page_config(
    page_title="Traffic Analysis Tool · Lima",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Activar sistema de diseño completo
apply_theme()

# Logger
logger = setup_logger('app', 'logs/app.log')


# ==================== ESTADO DE LA SESIÓN ====================

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.video_uploaded = False
    st.session_state.video_path = None
    st.session_state.video_properties = None
    st.session_state.detector = None
    st.session_state.tracker = None
    st.session_state.counter = None
    st.session_state.counting_lines = []
    st.session_state.processing_complete = False
    st.session_state.statistics = None
    st.session_state.crossings = []
    st.session_state.time_series = None
    st.session_state.output_dirs = None
    st.session_state.frame_for_lines = None
    st.session_state.processed_video_path = None


# ==================== FUNCIONES AUXILIARES ====================

def load_video(video_file):
    """Carga y guarda el video temporalmente."""
    try:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, video_file.name)
        with open(video_path, 'wb') as f:
            f.write(video_file.read())
        if not validate_video_file(video_path):
            st.error("El archivo no es un video válido.")
            return None
        properties = get_video_properties(video_path)
        return video_path, properties
    except Exception as e:
        st.error(f"Error al cargar video: {e}")
        logger.error(f"Error al cargar video: {e}")
        return None


def get_first_frame(video_path):
    """Obtiene el primer frame como RGB uint8 contiguo (seguro para st.image)."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    return None


def frame_to_pil(frame_rgb):
    """Convierte array RGB uint8 → PIL Image. Siempre seguro para st.image."""
    arr = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    from PIL import Image as PILImage
    return PILImage.fromarray(arr)


def draw_line_on_frame(frame_rgb, lines):
    """
    Dibuja líneas sobre frame RGB. Trabaja en BGR internamente y devuelve RGB uint8.
    """
    bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb, dtype=np.uint8),
                       cv2.COLOR_RGB2BGR)
    for line in lines:
        cv2.line(bgr, line['start'], line['end'], line['color'], 3)
        mid_x = (line['start'][0] + line['end'][0]) // 2
        mid_y = (line['start'][1] + line['end'][1]) // 2
        cv2.putText(bgr, line['name'], (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line['color'], 2)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _step_actual():
    """Calcula en qué paso del flujo está el usuario."""
    if st.session_state.processing_complete:
        return 3
    if len(st.session_state.counting_lines) > 0:
        return 2
    if st.session_state.detector is not None:
        return 1
    if st.session_state.video_uploaded:
        return 1
    return 0


def process_video(video_path, detector, tracker, counter,
                  save_video=False, show_visualizations=True,
                  use_opencv_window=False):
    """
    Procesa el video completo: detección -> tracking -> conteo.
    Devuelve (success, output_video_path, statistics, crossings, time_series).
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = 2  # ← aquí

        output_video_path = None
        if save_video:
            output_dirs = st.session_state.output_dirs
            output_video_path = os.path.join(
                output_dirs['videos'],
                f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if show_visualizations:
            progress_bar      = st.progress(0)
            status_text       = st.empty()
            metrics_slot      = st.empty()
            video_placeholder = st.empty()
            stats_placeholder = st.empty()

        frame_count   = 0
        all_crossings = []
        proc_start    = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            detections   = detector.detect(frame)
            tracks       = tracker.update(detections, frame_count)
            valid_tracks = tracker.get_valid_tracks()
            crossings    = counter.update(valid_tracks, frame_count)
            all_crossings.extend(crossings)

            if save_video or show_visualizations:
                frame_vis = frame.copy()
                for track in valid_tracks:
                    bbox  = track.get_current_bbox()
                    color = OBJECT_CLASSES.get(track.class_name, {}).get('color', (200, 200, 200))
                    x1, y1, x2, y2 = bbox

                    # Box con esquinas redondeadas visualmente
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)

                    # Fondo semitransparente para el label
                    label     = f"{track.class_spanish} #{track.track_id}"
                    font      = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.45
                    thickness  = 1
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    # Fondo del label
                    cv2.rectangle(frame_vis,
                                  (x1, y1 - th - 8),
                                  (x1 + tw + 6, y1),
                                  color, -1)
                    # Texto en negro sobre el fondo de color
                    cv2.putText(frame_vis, label,
                                (x1 + 3, y1 - 4),
                                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                    # Trayectoria con fade (más reciente = más opaco)
                    traj = track.get_trajectory()
                    if len(traj) > 1:
                        for i in range(len(traj) - 1):
                            alpha = (i + 1) / len(traj)
                            t_color = tuple(int(c * alpha) for c in color)
                            cv2.line(frame_vis, traj[i], traj[i+1], t_color, 1)

                frame_vis = counter.draw_lines(frame_vis, show_counts=True)

                # Overlay de FPS en esquina superior derecha
                elapsed_now = time.time() - proc_start
                live_fps    = frame_count / max(elapsed_now, 0.001)
                cv2.putText(frame_vis,
                            f"FPS: {live_fps:.1f}  Frame: {frame_count}/{total_frames}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)

                if save_video:
                    out.write(frame_vis)

                # Ventana OpenCV fluida — se actualiza cada frame
                if show_visualizations and use_opencv_window:
                    cv2.imshow("Traffic Analysis — procesando (Q para cerrar)", frame_vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if show_visualizations and frame_count % 10 == 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                elapsed  = frame_count / fps
                status_text.text(
                    f"Frame {frame_count}/{total_frames}  "
                    f"({progress*100:.1f}%)  —  {elapsed:.1f}s procesados"
                )

                current_stats = counter.get_statistics()
                total_counts  = current_stats.get('total_counts', {})

                if total_counts:
                    counts_es = {}
                    for cls_en, cnt in total_counts.items():
                        cls_es = OBJECT_CLASSES.get(cls_en, {}).get('name', cls_en)
                        counts_es[cls_es] = cnt
                    proc_fps = frame_count / max(time.time() - proc_start, 0.001)
                    with metrics_slot.container():
                        live_metrics_panel(
                            counts=counts_es,
                            fps=proc_fps,
                            elapsed=time.time() - proc_start
                        )

                if frame_count % 30 == 0:
                    frame_rgb = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb,
                                           caption=f"Frame {frame_count}",
                                           use_column_width=True)

                if frame_count % 60 == 0:
                    with stats_placeholder.container():
                        section_title("Estadísticas en tiempo real")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Cruces registrados", current_stats['total_crossings'])
                        c2.metric("Tracks activos",     len(valid_tracks))
                        c3.metric("Tiempo procesado",   f"{elapsed:.1f}s")

        cap.release()
        if save_video:
            out.release()
        # Cerrar ventana OpenCV al terminar
        if use_opencv_window:
            cv2.destroyAllWindows()

        statistics  = counter.get_statistics()
        time_series = counter.get_time_series_data(interval_seconds=60)

        if show_visualizations:
            progress_bar.progress(1.0)
            status_text.text("Procesamiento completado.")

        return True, output_video_path, statistics, all_crossings, time_series

    except Exception as e:
        logger.error(f"Error en procesamiento: {e}")
        st.error(f"Error durante el procesamiento: {e}")
        return False, None, None, [], None


# ==================== INTERFAZ PRINCIPAL ====================

def main():

    # ── HEADER ────────────────────────────────────────────────
    header(
        title="Traffic Analysis Tool",
        subtitle="Análisis vehicular con YOLOv8 · Lima, Perú",
        badge="v1.0 · YOLOv8n"
    )

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        sidebar_brand()

        step_indicator(
            steps=["Cargar video", "Inicializar detector",
                   "Definir líneas", "Procesar"],
            current=_step_actual()
        )

        st.markdown("---")

        # ── Sección 1: Video ──────────────────────────────────
        sidebar_label("1 · Video")

        video_file = st.file_uploader(
            "Selecciona un video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="MP4, AVI, MOV o MKV"
        )

        if video_file is not None and not st.session_state.video_uploaded:
            with st.spinner("Cargando video..."):
                result = load_video(video_file)
                if result:
                    st.session_state.video_path, st.session_state.video_properties = result
                    st.session_state.video_uploaded = True
                    st.session_state.frame_for_lines = get_first_frame(
                        st.session_state.video_path)
                    st.session_state.output_dirs = create_output_dirs()
                    st.success("Video cargado.")

        if st.session_state.video_uploaded:
            video_info_card(st.session_state.video_properties)
            est = estimate_processing_time(st.session_state.video_path)
            st.caption(f"Tiempo estimado: {est/60:.1f} min")

        st.markdown("---")

        # ── Sección 2: Detector ───────────────────────────────
        sidebar_label("2 · Detector")

        confidence = st.slider(
            "Confianza mínima",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
            help="Mayor valor = menos detecciones pero más precisas"
        )

        model_size = st.selectbox(
            "Tamaño del modelo",
            options=['n', 's', 'm', 'l'],
            index=0,
            help="n=nano (rápido)  s=small  m=medium  l=large (preciso)"
        )

        sidebar_label("Clases a detectar")
        available_classes   = list(OBJECT_CLASSES.keys())
        class_names_spanish = [OBJECT_CLASSES[c]['name'] for c in available_classes]
        selected_classes_es = st.multiselect(
            "Categorías",
            options=class_names_spanish,
            default=class_names_spanish,
        )
        selected_classes = [k for k, v in OBJECT_CLASSES.items()
                            if v['name'] in selected_classes_es]
        use_peruvian = st.checkbox(
        "Usar modelo peruano",
        value=True,
        help="vehiculos_lima_v1.pt — entrenado con clases de Lima"
        )                    

        det_status = "ready" if st.session_state.detector else "inactive"
        status_badge(
            "Detector listo" if st.session_state.detector else "Sin inicializar",
            det_status
        )

        if st.button("Inicializar detector", type="primary"):
            with st.spinner("Inicializando..."):
                try:
                    st.session_state.detector = create_detector(
                        confidence=confidence,
                        model_size=model_size,
                        classes=selected_classes if selected_classes else None,
                        use_peruvian_model=use_peruvian  # ← agregar esta línea
                    )
                    st.session_state.tracker = ObjectTracker(
                        max_age=10, min_hits=3, iou_threshold=0.3
                    )
                    fps_vid = (st.session_state.video_properties['fps']
                               if st.session_state.video_properties else 30)
                    st.session_state.counter = TrafficCounter(fps=fps_vid)
                    st.success("Detector y tracker listos.")
                    logger.info("Detector inicializado correctamente")
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Error al inicializar detector: {e}")

        st.markdown("---")

        with st.expander("Acerca de"):
            st.caption(
                "Traffic Analysis Tool v1.0\n\n"
                "YOLOv8 · Tracking multi-objeto · "
                "Líneas de conteo · Excel / CSV\n\n"
                "Streamlit + OpenCV + PyTorch"
            )

    # ==================== TABS PRINCIPALES ====================

    tab1, tab2, tab3, tab4 = st.tabs([
        "Líneas", "Procesar", "Resultados", "Exportar"
    ])

    # ── TAB 1: Definir Líneas ─────────────────────────────────
    with tab1:
        section_title("Definir líneas de conteo")

        if not st.session_state.video_uploaded:
            empty_state(
                icon="📹",
                title="Sin video cargado",
                description="Sube un video desde el sidebar para definir las líneas de conteo."
            )
        else:
            col1, col2 = st.columns([3, 2], gap="large")

            # ── Columna izquierda: visor del frame ────────────
            with col1:
                # Cabecera del visor con info de resolución
                props = st.session_state.video_properties or {}
                w, h  = props.get('width', '—'), props.get('height', '—')
                n_lines = len(st.session_state.counting_lines)
                st.markdown(f"""
                <div style="
                    display:flex; align-items:center; justify-content:space-between;
                    background:#181C24; border:1px solid #1E2433;
                    border-radius:10px 10px 0 0; padding:10px 16px;
                    border-bottom: none;
                ">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:#22C55E;display:inline-block;"></span>
                        <span style="font-family:'Sora',sans-serif;font-size:0.78rem;
                                     font-weight:600;color:#94A3B8;letter-spacing:0.04em;">
                            FRAME DE REFERENCIA
                        </span>
                    </div>
                    <div style="display:flex;gap:10px;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                     color:#475569;background:#111318;border:1px solid #1E2433;
                                     border-radius:5px;padding:2px 8px;">{w}×{h}</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                     color:#F5A623;background:rgba(245,166,35,0.1);border:1px solid #7A5210;
                                     border-radius:5px;padding:2px 8px;">{n_lines} línea{'s' if n_lines!=1 else ''}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Frame
                if st.session_state.frame_for_lines is not None:
                    frame_display = st.session_state.frame_for_lines.copy()
                    if st.session_state.counting_lines:
                        frame_display = draw_line_on_frame(
                            frame_display, st.session_state.counting_lines)
                    st.markdown("""
                    <div style="border:1px solid #1E2433;border-top:none;
                                border-radius:0 0 10px 10px;overflow:hidden;line-height:0;">
                    """, unsafe_allow_html=True)
                    st.image(frame_to_pil(frame_display), use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Tip de coordenadas
                st.markdown("""
                <div style="margin-top:10px;background:#111318;border:1px solid #1E2433;
                             border-radius:8px;padding:10px 14px;display:flex;gap:8px;align-items:flex-start;">
                    <span style="color:#F5A623;font-size:0.85rem;margin-top:1px;">💡</span>
                    <span style="font-family:'Sora',sans-serif;font-size:0.78rem;color:#475569;line-height:1.5;">
                        Las coordenadas (0,0) están en la esquina <strong style="color:#94A3B8;">superior izquierda</strong>.
                        X crece hacia la derecha, Y hacia abajo.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # ── Columna derecha: formulario + lista ───────────
            with col2:
                # Formulario
                st.markdown("""
                <div style="background:#111318;border:1px solid #1E2433;
                             border-radius:12px;padding:18px 18px 4px 18px;margin-bottom:16px;">
                    <div style="font-family:'Sora',sans-serif;font-size:0.7rem;font-weight:600;
                                letter-spacing:0.12em;text-transform:uppercase;color:#475569;
                                margin-bottom:14px;display:flex;align-items:center;gap:8px;">
                        <span style="width:3px;height:12px;background:#F5A623;border-radius:2px;display:inline-block;"></span>
                        Nueva línea
                    </div>
                """, unsafe_allow_html=True)

                with st.form("add_line_form"):
                    line_name = st.text_input(
                        "Nombre de la línea",
                        value=f"Línea {len(st.session_state.counting_lines)+1}",
                        placeholder="Ej: Norte–Sur, Carril 1…"
                    )
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption("Punto inicio")
                        start_x = st.number_input("X₁", min_value=0, value=100, label_visibility="collapsed")
                        start_y = st.number_input("Y₁", min_value=0, value=300, label_visibility="collapsed")
                        st.caption("X  ·  Y")
                    with col_b:
                        st.caption("Punto fin")
                        end_x = st.number_input("X₂", min_value=0, value=500, label_visibility="collapsed")
                        end_y = st.number_input("Y₂", min_value=0, value=300, label_visibility="collapsed")
                        st.caption("X  ·  Y")

                    line_color = st.color_picker("Color de la línea", value="#F5A623")

                    if st.form_submit_button("＋  Agregar línea", type="primary"):
                        hex_c = line_color.lstrip('#')
                        rgb   = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
                        bgr   = (rgb[2], rgb[1], rgb[0])
                        st.session_state.counting_lines.append({
                            'name':  line_name,
                            'start': (start_x, start_y),
                            'end':   (end_x,   end_y),
                            'color': bgr
                        })
                        st.success(f"✓  Línea '{line_name}' agregada.")
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

                # Lista de líneas definidas
                n = len(st.session_state.counting_lines)
                section_title(f"Líneas definidas · {n}")

                if st.session_state.counting_lines:
                    LINE_COLORS_HEX = {
                        'car': '#3B82F6', 'moto': '#F5A623', 'bus': '#22C55E',
                        'truck': '#EF4444', 'default': '#94A3B8'
                    }
                    for idx, line in enumerate(st.session_state.counting_lines):
                        # Convertir color BGR → hex para la pastilla
                        bgr = line['color']
                        hex_color = '#{:02x}{:02x}{:02x}'.format(bgr[2], bgr[1], bgr[0])
                        c_info, c_del = st.columns([5, 1])
                        with c_info:
                            st.markdown(f"""
                            <div style="background:#111318;border:1px solid #1E2433;
                                        border-left:3px solid {hex_color};
                                        border-radius:8px;padding:10px 14px;margin-bottom:6px;">
                                <div style="font-family:'Sora',sans-serif;font-size:0.84rem;
                                            font-weight:600;color:#F1F5F9;margin-bottom:4px;">
                                    {line['name']}
                                </div>
                                <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                            color:#475569;">
                                    ({line['start'][0]}, {line['start'][1]})
                                    <span style="color:#2A3347;margin:0 4px;">→</span>
                                    ({line['end'][0]}, {line['end'][1]})
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with c_del:
                            st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
                            if st.button("✕", key=f"del_{idx}"):
                                st.session_state.counting_lines.pop(idx)
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    empty_state(
                        icon="📏",
                        title="Sin líneas",
                        description="Agrega al menos una línea para continuar."
                    )

    # ── TAB 2: Procesar ───────────────────────────────────────
    with tab2:
        section_title("Procesamiento de video")

        ok_video    = st.session_state.video_uploaded
        ok_detector = st.session_state.detector is not None
        ok_lines    = len(st.session_state.counting_lines) > 0

        # ── Checklist de requisitos ───────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            status_badge("Video cargado",  "ready" if ok_video    else "pending")
        with c2:
            status_badge("Detector listo", "ready" if ok_detector else "pending")
        with c3:
            status_badge("Líneas listas",  "ready" if ok_lines    else "pending")

        st.markdown("")

        if not (ok_video and ok_detector and ok_lines):
            empty_state(
                icon="⚙️",
                title="Configuración incompleta",
                description="Completa los 3 pasos del sidebar antes de procesar."
            )
        else:
            # ── Info del video + modo de procesamiento ────────
            vinfo       = get_video_info(st.session_state.video_path)
            dur_str     = format_duration(vinfo["duration_s"])
            use_chunks  = needs_chunking(st.session_state.video_path, threshold_minutes=30)
            has_ffmpeg  = ffmpeg_available()

            if use_chunks:
                chunker      = VideoChunker(st.session_state.video_path,
                                            chunk_minutes=15)
                n_chunks     = chunker.total_chunks
                chunk_info   = chunker.get_chunk_info()
                est_proc_min = n_chunks * 3   # ~3 min por chunk con yolov8n
            else:
                n_chunks   = 1
                chunk_info = []

            # Panel informativo del video
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1E2433;border-radius:12px;
                        padding:16px 20px;margin-bottom:16px;">
                <div style="display:flex;align-items:center;justify-content:space-between;
                            flex-wrap:wrap;gap:12px;">
                    <div style="display:flex;align-items:center;gap:14px;">
                        <div style="font-size:1.8rem;opacity:0.7;">🎬</div>
                        <div>
                            <div style="font-family:'Sora',sans-serif;font-size:0.84rem;
                                        font-weight:600;color:#F1F5F9;">
                                {os.path.basename(st.session_state.video_path).split('_')[-1] if '_' in os.path.basename(st.session_state.video_path) else 'video.mp4'}
                            </div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                        color:#475569;margin-top:3px;">
                                {vinfo["width"]}×{vinfo["height"]}  ·  {vinfo["fps"]:.0f}fps  ·  {vinfo["size_mb"]:.0f}MB
                            </div>
                        </div>
                    </div>
                    <div style="display:flex;gap:10px;flex-wrap:wrap;">
                        <div style="background:#0A0C10;border:1px solid #1E2433;border-radius:8px;
                                    padding:8px 14px;text-align:center;">
                            <div style="font-family:'Sora',sans-serif;font-size:0.62rem;
                                        font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
                                        color:#475569;margin-bottom:2px;">Duración</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                                        font-weight:500;color:#F5A623;">{dur_str}</div>
                        </div>
                        <div style="background:#0A0C10;border:1px solid #1E2433;border-radius:8px;
                                    padding:8px 14px;text-align:center;">
                            <div style="font-family:'Sora',sans-serif;font-size:0.62rem;
                                        font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
                                        color:#475569;margin-bottom:2px;">Modo</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                                        font-weight:500;color:{'#22C55E' if use_chunks else '#3B82F6'};">
                                {'Chunks ×' + str(n_chunks) if use_chunks else 'Directo'}
                            </div>
                        </div>
                        <div style="background:#0A0C10;border:1px solid #1E2433;border-radius:8px;
                                    padding:8px 14px;text-align:center;">
                            <div style="font-family:'Sora',sans-serif;font-size:0.62rem;
                                        font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
                                        color:#475569;margin-bottom:2px;">Est. proceso</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                                        font-weight:500;color:#94A3B8;">
                                ~{est_proc_min if use_chunks else int(vinfo["duration_min"] * 0.3)} min
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Aviso si video largo pero sin ffmpeg
            if use_chunks and not has_ffmpeg:
                st.markdown(f"""
                <div style="background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.3);
                             border-radius:10px;padding:14px 18px;margin-bottom:12px;">
                    <div style="font-family:'Sora',sans-serif;font-size:0.84rem;
                                font-weight:600;color:#F5A623;margin-bottom:6px;">
                        ⚠️  Video largo detectado — ffmpeg no está instalado
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                                color:#94A3B8;line-height:1.6;">
                        El video dura {dur_str}. Para videos &gt;30 min se recomienda
                        procesamiento por chunks. Instala ffmpeg para activarlo:<br>
                        <span style="color:#F5A623;">winget install ffmpeg</span>
                        &nbsp;·&nbsp; luego reinicia la app.
                        <br>Por ahora se procesará completo (puede tardar más).
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Plan de chunks (expandible)
            if use_chunks and has_ffmpeg and chunk_info:
                with st.expander(f"Ver plan de procesamiento — {n_chunks} chunks de 15 min"):
                    rows_html = "".join([
                        f"""<div style="display:flex;justify-content:space-between;
                                        padding:6px 0;border-bottom:1px solid #1E2433;
                                        font-family:'JetBrains Mono',monospace;font-size:0.75rem;">
                                <span style="color:#475569;">Chunk {ci['index']+1:02d}</span>
                                <span style="color:#94A3B8;">{ci['start_str']} → {ci['end_str']}</span>
                                <span style="color:#475569;">{format_duration(ci['duration_s'])}</span>
                            </div>"""
                        for ci in chunk_info
                    ])
                    st.markdown(f"""
                    <div style="background:#0A0C10;border:1px solid #1E2433;border-radius:8px;
                                padding:12px 16px;max-height:200px;overflow-y:auto;">
                        {rows_html}
                    </div>
                    """, unsafe_allow_html=True)

            # Opciones
            col1, col2, col3 = st.columns(3)
            with col1:
                save_video = st.checkbox(
                    "Guardar video procesado", value=False,
                    help="Guarda el video con anotaciones — requiere espacio extra en disco"
                )
            with col2:
                show_viz = st.checkbox(
                    "Visualización en tiempo real", value=True,
                    help="Muestra frame y métricas durante el proceso"
                )
            with col3:
                use_opencv_window = st.checkbox(
                    "Ventana fluida (OpenCV)", value=False,
                    help="Abre una ventana externa con el video en tiempo real. Presiona Q para cerrar."
                )

            st.markdown("")

            if st.button("Iniciar procesamiento", type="primary", use_container_width=True):
                section_title("Procesando…")

                # Registrar líneas en el counter
                for line in st.session_state.counting_lines:
                    st.session_state.counter.add_line(
                        start=line['start'], end=line['end'],
                        name=line['name'],   color=line['color']
                    )

                t0 = time.time()

                # ══ MODO CHUNKS ══════════════════════════════
                if use_chunks and has_ffmpeg:
                    st.markdown(f"""
                    <div style="background:#111318;border:1px solid #1E2433;border-radius:10px;
                                padding:12px 16px;margin-bottom:12px;">
                        <span style="font-family:'Sora',sans-serif;font-size:0.78rem;
                                     color:#94A3B8;">
                            Partiendo video en {n_chunks} segmentos de 15 min con ffmpeg…
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    chunk_bar     = st.progress(0)
                    chunk_status  = st.empty()
                    all_stats     = []
                    all_crossings = []
                    failed_chunks = []

                    try:
                        chunker = VideoChunker(
                            st.session_state.video_path,
                            chunk_minutes=15,
                            output_dir=os.path.join(
                                st.session_state.output_dirs['base'], "chunks"
                            )
                        )
                        chunk_paths = chunker.split()

                        for i, chunk_path in enumerate(chunk_paths):
                            lbl = chunker.chunk_label(i)
                            chunk_status.markdown(f"""
                            <div style="font-family:'JetBrains Mono',monospace;
                                        font-size:0.8rem;color:#F5A623;padding:4px 0;">
                                ▶ {lbl}
                            </div>
                            """, unsafe_allow_html=True)

                            success, _, stats, crossings, time_series = process_video(
                                chunk_path,
                                st.session_state.detector,
                                st.session_state.tracker,
                                st.session_state.counter,
                                save_video=False,
                                show_visualizations=show_viz,
                                use_opencv_window=use_opencv_window
                            )

                            if success and stats:
                                all_stats.append(stats)
                                all_crossings.extend(crossings)
                            else:
                                failed_chunks.append(lbl)

                            chunker.delete_chunk(chunk_path)  # libera disco
                            chunk_bar.progress((i + 1) / len(chunk_paths))

                        chunk_status.empty()
                        chunk_bar.progress(1.0)

                        # Consolidar todo
                        final_stats     = consolidate_stats(all_stats)
                        final_crossings = consolidate_crossings(
                            [all_crossings]
                        )

                        proc_time = time.time() - t0
                        st.session_state.processing_complete  = True
                        st.session_state.statistics           = final_stats
                        st.session_state.crossings            = final_crossings
                        st.session_state.time_series          = None
                        st.session_state.processed_video_path = None

                        if failed_chunks:
                            st.warning(
                                f"Completado con {len(failed_chunks)} chunks fallidos: "
                                f"{', '.join(failed_chunks)}"
                            )
                        else:
                            st.success(
                                f"✓ {n_chunks} chunks procesados en "
                                f"{proc_time/60:.1f} min."
                            )

                    except Exception as e:
                        st.error(f"Error en procesamiento por chunks: {e}")
                        logger.error(f"Error chunks: {e}")

                # ══ MODO DIRECTO ══════════════════════════════
                else:
                    success, vid_path, stats, crossings, time_series = process_video(
                        st.session_state.video_path,
                        st.session_state.detector,
                        st.session_state.tracker,
                        st.session_state.counter,
                        save_video=save_video,
                        show_visualizations=show_viz,
                        use_opencv_window=use_opencv_window
                    )
                    proc_time = time.time() - t0

                    if success:
                        st.session_state.processing_complete  = True
                        st.session_state.statistics           = stats
                        st.session_state.crossings            = crossings
                        st.session_state.time_series          = time_series
                        st.session_state.processed_video_path = vid_path
                        st.success(f"✓ Completado en {proc_time/60:.1f} min.")
                    else:
                        st.error("Error durante el procesamiento. Revisa los logs.")

                # ══ RESUMEN RÁPIDO (ambos modos) ══════════════
                if st.session_state.processing_complete:
                    s = st.session_state.statistics
                    section_title("Resumen rápido")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Cruces totales",   s.get('total_crossings', 0))
                    r2.metric("Líneas usadas",    s.get('total_lines', 0))
                    r3.metric("Tiempo analizado", f"{s.get('elapsed_minutes',0):.1f} min")
                    r4.metric("Objetos únicos",   sum(s.get('total_counts',{}).values()))

                    if (st.session_state.processed_video_path and
                            os.path.exists(st.session_state.processed_video_path)):
                        section_title("Video procesado")
                        video_player(st.session_state.processed_video_path, height=400)

    # ── TAB 3: Resultados ─────────────────────────────────────
    with tab3:
        section_title("Resultados del análisis")

        if not st.session_state.processing_complete:
            empty_state(
                icon="📊",
                title="Sin resultados aún",
                description="Procesa un video primero para ver los resultados aquí."
            )
        else:
            stats     = st.session_state.statistics
            crossings = st.session_state.crossings

            section_title("Resumen general")
            total = sum(stats['total_counts'].values())

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                metric_card("Cruces totales",   stats['total_crossings'],
                            subtitle="eventos registrados")
            with kpi2:
                metric_card("Líneas activas",   stats['total_lines'],
                            subtitle="líneas de conteo")
            with kpi3:
                metric_card("Tiempo analizado", f"{stats['elapsed_minutes']:.1f} min",
                            subtitle="duración del video")
            with kpi4:
                metric_card("Objetos únicos",   total,
                            subtitle="todos los tipos")

            st.markdown("")

            section_title("Conteos por categoría")
            total_counts = stats['total_counts']

            if total_counts:
                CLASS_VEHICLE_MAP = {
                    'Auto': 'car', 'Moto': 'moto', 'Bus': 'bus',
                    'Camión': 'truck', 'Peatón': 'person', 'Bicicleta': 'bicycle',
                }
                cols = st.columns(len(total_counts))
                for i, (cls_en, count) in enumerate(total_counts.items()):
                    cls_es = OBJECT_CLASSES.get(cls_en, {}).get('name', cls_en)
                    v_cls  = CLASS_VEHICLE_MAP.get(cls_es, '')
                    total_all = sum(total_counts.values())
                    pct = f"{count/total_all*100:.0f}%" if total_all > 0 else "—"
                    with cols[i]:
                        metric_card(cls_es, count, subtitle=pct, vehicle_class=v_cls)

                st.markdown("")
                df_counts = pd.DataFrame({
                    'Categoría': [OBJECT_CLASSES.get(k,{}).get('name',k)
                                  for k in total_counts.keys()],
                    'Conteo': list(total_counts.values())
                })
                st.bar_chart(df_counts.set_index('Categoría'))

            st.markdown("---")

            section_title("Conteos por línea")
            for line_info in stats['counts_by_line']:
                with st.expander(
                    f"{line_info['line_name']}  —  total: {line_info['total_count']}"
                ):
                    rows = []
                    for class_name, directions in line_info['counts_by_class'].items():
                        rows.append({
                            'Categoría':    class_name,
                            'Total':        directions['total'],
                            'Arriba→Abajo': directions.get('up_to_down',    0),
                            'Abajo→Arriba': directions.get('down_to_up',    0),
                            'Izq→Der':      directions.get('left_to_right', 0),
                            'Der→Izq':      directions.get('right_to_left', 0),
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows),
                                     use_container_width=True, hide_index=True)

            st.markdown("---")

            section_title("Detalle de cruces")
            if crossings:
                df_crossings = pd.DataFrame([
                    {
                        'Timestamp': f"{c['timestamp']:.2f}s",
                        'Track ID':  c['track_id'],
                        'Categoría': c['class_spanish'],
                        'Línea':     c['line_name'],
                        'Dirección': c['direction'],
                    }
                    for c in crossings[:100]
                ])
                st.dataframe(df_crossings, use_container_width=True, hide_index=True)
                if len(crossings) > 100:
                    st.caption(f"Mostrando 100 de {len(crossings)} cruces totales.")
            else:
                st.info("No se detectaron cruces.")

            st.markdown("---")

            if st.button("Generar visualizaciones", type="primary"):
                with st.spinner("Generando gráficos..."):
                    visualizer = TrafficVisualizer(
                        output_dir=st.session_state.output_dirs['visualizations']
                    )
                    frame_shape = (
                        st.session_state.video_properties['height'],
                        st.session_state.video_properties['width']
                    )
                    viz_files = visualizer.generate_all_visualizations(
                        stats, crossings,
                        st.session_state.time_series, frame_shape
                    )
                    st.success(f"{len(viz_files)} visualizaciones generadas.")
                    for viz_name, viz_path in viz_files.items():
                        if os.path.exists(viz_path):
                            st.image(viz_path, caption=viz_name,
                                     use_column_width=True)

    # ── TAB 4: Exportar ───────────────────────────────────────
    with tab4:
        section_title("Exportar resultados")

        if not st.session_state.processing_complete:
            empty_state(
                icon="📥",
                title="Sin datos para exportar",
                description="Procesa un video primero para habilitar la exportación."
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                section_title("Excel completo")
                st.caption(
                    "Resumen general · conteos totales · conteos por línea · "
                    "detalle de cruces · series temporales · estadísticas por dirección."
                )
            with col2:
                section_title("CSV datos crudos")
                st.caption(
                    "CSV de cruces completos · CSV de resumen · CSV de series temporales."
                )

            st.markdown("---")

            btn1, btn2, btn3 = st.columns(3)

            with btn1:
                if st.button("Exportar a Excel", type="primary", use_container_width=True):
                    with st.spinner("Generando Excel..."):
                        exporter = DataExporter(
                            output_dir=st.session_state.output_dirs['exports'],
                            video_name="traffic_analysis"
                        )
                        excel_path = exporter.export_to_excel(
                            st.session_state.statistics,
                            st.session_state.crossings,
                            st.session_state.time_series,
                            st.session_state.video_properties
                        )
                        st.success("Excel generado.")
                        with open(excel_path, 'rb') as f:
                            st.download_button(
                                label="Descargar Excel", data=f,
                                file_name=os.path.basename(excel_path),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

            with btn2:
                if st.button("Exportar CSV", type="primary", use_container_width=True):
                    with st.spinner("Generando CSV..."):
                        exporter = DataExporter(
                            output_dir=st.session_state.output_dirs['exports'],
                            video_name="traffic_analysis"
                        )
                        csv_path = exporter.export_to_csv(st.session_state.crossings)
                        st.success("CSV generado.")
                        with open(csv_path, 'rb') as f:
                            st.download_button(
                                label="Descargar CSV", data=f,
                                file_name=os.path.basename(csv_path),
                                mime="text/csv"
                            )

            with btn3:
                if st.button("Exportar todo", type="primary", use_container_width=True):
                    with st.spinner("Generando archivos..."):
                        exporter = DataExporter(
                            output_dir=st.session_state.output_dirs['exports'],
                            video_name="traffic_analysis"
                        )
                        files = exporter.export_all(
                            st.session_state.statistics,
                            st.session_state.crossings,
                            st.session_state.time_series,
                            st.session_state.video_properties
                        )
                        st.success(f"{len(files)} archivos generados.")

            st.markdown("---")

            if st.session_state.output_dirs:
                section_title("Archivos generados")
                st.code(st.session_state.output_dirs['base'])
                for subdir in ['exports', 'visualizations', 'videos']:
                    dir_path = st.session_state.output_dirs[subdir]
                    if os.path.exists(dir_path):
                        files = os.listdir(dir_path)
                        if files:
                            with st.expander(f"{subdir.upper()}  ({len(files)})"):
                                for f in files:
                                    st.text(f"  {f}")


if __name__ == "__main__":
    main()