"""
components.py - Componentes HTML/JS embebidos para Traffic Analysis Tool

Usa st.components.v1.html() para renderizar elementos que Streamlit
no puede construir nativamente: reproductor de video con controles,
panel de métricas animado, canvas de zonas OD (Fase 4).

Importar desde app.py:
    from components import video_player, metrics_live_panel, od_zone_canvas
"""

import streamlit as st
import streamlit.components.v1 as components
from theme import COLORS


# ─────────────────────────────────────────────
# REPRODUCTOR DE VIDEO MEJORADO
# ─────────────────────────────────────────────

def video_player(video_path: str, height: int = 400):
    """
    Reproductor HTML5 con controles nativos mejorados y soporte
    para el video procesado exportado.

    Args:
        video_path: Ruta local o URL al archivo de video
        height:     Alto del reproductor en px
    """
    c = COLORS
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Barlow+Condensed:wght@600&display=swap');

        .vp-wrap {{
            background: {c['bg_surface']};
            border: 1px solid {c['border']};
            border-radius: 10px;
            overflow: hidden;
            font-family: 'Barlow Condensed', sans-serif;
        }}
        .vp-video {{
            width: 100%;
            height: {height}px;
            object-fit: contain;
            background: #000;
            display: block;
        }}
        .vp-controls {{
            background: {c['bg_raised']};
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 12px;
            border-top: 1px solid {c['border']};
        }}
        .vp-btn {{
            background: transparent;
            border: 1px solid {c['border_bright']};
            border-radius: 5px;
            color: {c['text_primary']};
            width: 34px;
            height: 30px;
            cursor: pointer;
            font-size: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
            flex-shrink: 0;
        }}
        .vp-btn:hover {{
            border-color: {c['amber']};
            color: {c['amber']};
        }}
        .vp-seek {{
            flex: 1;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: {c['border_bright']};
            border-radius: 2px;
            outline: none;
            cursor: pointer;
        }}
        .vp-seek::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            background: {c['amber']};
            border-radius: 50%;
            cursor: pointer;
        }}
        .vp-time {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: {c['text_secondary']};
            white-space: nowrap;
            min-width: 90px;
            text-align: right;
        }}
        .vp-zoom-wrap {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.72rem;
            color: {c['text_muted']};
        }}
        .vp-zoom-wrap input {{
            width: 60px;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: {c['border_bright']};
            border-radius: 2px;
            outline: none;
            cursor: pointer;
        }}
        .vp-zoom-wrap input::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 10px;
            height: 10px;
            background: {c['amber']};
            border-radius: 50%;
        }}
    </style>

    <div class="vp-wrap">
        <div style="overflow:hidden; position:relative;">
            <video class="vp-video" id="vpVid" preload="metadata">
                <source src="{video_path}">
                Tu navegador no soporta video HTML5.
            </video>
        </div>
        <div class="vp-controls">
            <button class="vp-btn" id="vpPlay" title="Play / Pause">▶</button>
            <button class="vp-btn" id="vpRewind" title="−10s">↩</button>
            <input class="vp-seek" type="range" id="vpSeek" min="0" step="0.1" value="0">
            <span class="vp-time" id="vpTime">0:00 / 0:00</span>
            <div class="vp-zoom-wrap">
                🔍
                <input type="range" id="vpZoom" min="1" max="3" step="0.1" value="1" title="Zoom">
            </div>
        </div>
    </div>

    <script>
        const vid    = document.getElementById('vpVid');
        const btnPlay= document.getElementById('vpPlay');
        const seek   = document.getElementById('vpSeek');
        const timeEl = document.getElementById('vpTime');
        const zoom   = document.getElementById('vpZoom');

        function fmt(s) {{
            const m = Math.floor(s/60), ss = Math.floor(s%60);
            return m + ':' + String(ss).padStart(2,'0');
        }}

        vid.addEventListener('loadedmetadata', () => {{
            seek.max = vid.duration;
            timeEl.textContent = '0:00 / ' + fmt(vid.duration);
        }});

        vid.addEventListener('timeupdate', () => {{
            seek.value = vid.currentTime;
            timeEl.textContent = fmt(vid.currentTime) + ' / ' + fmt(vid.duration);
        }});

        vid.addEventListener('play',  () => btnPlay.textContent = '⏸');
        vid.addEventListener('pause', () => btnPlay.textContent = '▶');
        vid.addEventListener('ended', () => btnPlay.textContent = '▶');

        btnPlay.addEventListener('click', () => {{
            vid.paused ? vid.play() : vid.pause();
        }});

        document.getElementById('vpRewind').addEventListener('click', () => {{
            vid.currentTime = Math.max(0, vid.currentTime - 10);
        }});

        seek.addEventListener('input', () => {{ vid.currentTime = seek.value; }});

        zoom.addEventListener('input', () => {{
            vid.style.transform = 'scale(' + zoom.value + ')';
            vid.style.transformOrigin = 'center center';
        }});
    </script>
    """
    components.html(html, height=height + 70, scrolling=False)


# ─────────────────────────────────────────────
# PANEL DE MÉTRICAS LIVE (animado)
# ─────────────────────────────────────────────

def metrics_live_panel(counts: dict, fps: float = 0.0, elapsed: float = 0.0):
    """
    Panel de métricas en tiempo real con animación de contadores.
    Se llama dentro del loop de procesamiento para actualizar los números.

    Args:
        counts:  {'Auto': 12, 'Moto': 5, 'Bus': 2, 'Camión': 1, 'Peatón': 8}
        fps:     FPS de procesamiento actual
        elapsed: Segundos transcurridos
    """
    c = COLORS
    CLASS_COLORS = {
        'Auto':      c['car'],
        'Moto':      c['moto'],
        'Bus':       c['bus'],
        'Camión':    c['red'],
        'Peatón':    '#A78BFA',
        'Bicicleta': '#06B6D4',
    }
    total = sum(counts.values())

    cards_html = ""
    for label, val in counts.items():
        color = CLASS_COLORS.get(label, c['border_bright'])
        pct = f"{val/total*100:.0f}%" if total > 0 else "0%"
        bar_w = f"{val/max(max(counts.values()),1)*100:.0f}%"
        cards_html += f"""
        <div style="
            background:{c['bg_surface']};
            border:1px solid {c['border']};
            border-top:2px solid {color};
            border-radius:8px;
            padding:12px 14px;
            min-width:110px;
        ">
            <p style="font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;
                      font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                      color:{c['text_muted']};margin:0 0 4px 0;">{label}</p>
            <p style="font-family:'IBM Plex Mono',monospace;font-size:1.7rem;
                      font-weight:500;color:{c['text_primary']};line-height:1;margin:0;">{val:,}</p>
            <div style="margin-top:6px;height:3px;background:{c['border']};border-radius:2px;">
                <div style="width:{bar_w};height:100%;background:{color};
                             border-radius:2px;transition:width 0.4s ease;"></div>
            </div>
            <p style="font-family:'Barlow',sans-serif;font-size:0.72rem;
                      color:{c['text_muted']};margin:3px 0 0 0;">{pct} del total</p>
        </div>"""

    # Tarjeta de totales y FPS
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    elapsed_str = f"{mins}:{str(secs).padStart if False else str(secs).zfill(2)}"

    cards_html += f"""
        <div style="
            background:{c['bg_raised']};
            border:1px solid {c['border_bright']};
            border-top:2px solid {c['amber']};
            border-radius:8px;
            padding:12px 14px;
            min-width:110px;
        ">
            <p style="font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;
                      font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                      color:{c['text_muted']};margin:0 0 4px 0;">Total</p>
            <p style="font-family:'IBM Plex Mono',monospace;font-size:1.7rem;
                      font-weight:500;color:{c['amber']};line-height:1;margin:0;">{total:,}</p>
            <p style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                      color:{c['text_secondary']};margin:6px 0 0 0;">
                ⚡ {fps:.1f} fps · ⏱ {elapsed_str}
            </p>
        </div>"""

    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Barlow+Condensed:wght@700&family=Barlow:wght@400&display=swap');
        .mlp-wrap {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            padding: 4px 0;
        }}
    </style>
    <div class="mlp-wrap">{cards_html}</div>
    """
    components.html(html, height=130, scrolling=False)


# ─────────────────────────────────────────────
# SIDEBAR LOGO / BRANDING
# ─────────────────────────────────────────────

def sidebar_brand():
    """Logo y nombre — st.markdown puro, sin iframe, fondo transparente."""
    st.markdown("""
    <div class="tat-brand">
        <div class="tat-brand-icon">🚦</div>
        <div>
            <div class="tat-brand-name">Traffic Analysis</div>
            <div class="tat-brand-ver">Lima · YOLOv8 · v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# EMPTY STATE (cuando no hay video cargado)
# ─────────────────────────────────────────────

def empty_state(icon: str, title: str, description: str):
    """Empty state — st.markdown puro, sin iframe, fondo transparente."""
    st.markdown(f"""
    <div class="tat-empty">
        <span class="tat-empty-icon">{icon}</span>
        <div class="tat-empty-title">{title}</div>
        <div class="tat-empty-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)