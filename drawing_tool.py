import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
import json

class DrawingTool:
    """Herramienta de dibujo interactivo para definir zonas y líneas de conteo."""
    
    def __init__(self, frame, canvas_key="canvas"):
        """
        Inicializa la herramienta de dibujo.
        
        Args:
            frame: Frame de video (numpy array BGR)
            canvas_key: Clave única para el canvas de Streamlit
        """
        self.frame = frame
        self.canvas_key = canvas_key
        self.height, self.width = frame.shape[:2]
        
        # Convertir frame BGR a RGB para Streamlit
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.background_image = Image.fromarray(self.frame_rgb)
    
    def draw_lines(self, existing_lines=None):
        """
        Interfaz para dibujar líneas de conteo.
        
        Args:
            existing_lines: Lista de líneas existentes para mostrar
            
        Returns:
            Lista de líneas dibujadas con formato:
            [{'name': str, 'direction': str, 'coords': [(x1,y1), (x2,y2)], 'color': str}]
        """
        st.subheader("📏 Dibujar Líneas de Conteo")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("**Configuración de Línea**")
            
            line_name = st.text_input(
                "Nombre de la Línea",
                value=f"Linea_{len(existing_lines) + 1 if existing_lines else 1}",
                key=f"line_name_{self.canvas_key}"
            )
            
            direction = st.selectbox(
                "Dirección del Flujo",
                ["N-S", "S-N", "E-O", "O-E", "NE-SO", "NO-SE", "SE-NO", "SO-NE"],
                key=f"direction_{self.canvas_key}"
            )
            
            line_color = st.color_picker(
                "Color de la Línea",
                "#00FF00",
                key=f"line_color_{self.canvas_key}"
            )
            
            stroke_width = st.slider(
                "Grosor de Línea",
                min_value=2,
                max_value=10,
                value=4,
                key=f"stroke_{self.canvas_key}"
            )
            
            st.info("💡 **Instrucciones:**\n\n"
                   "1. Selecciona la herramienta 'line' (📏)\n"
                   "2. Haz clic en el punto inicial\n"
                   "3. Arrastra hasta el punto final\n"
                   "4. Suelta para completar la línea")
        
        with col1:
            # Canvas para dibujar
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",  # Transparente
                stroke_width=stroke_width,
                stroke_color=line_color,
                background_image=self.background_image,
                update_streamlit=True,
                height=self.height,
                width=self.width,
                drawing_mode="line",
                point_display_radius=0,
                key=self.canvas_key,
                display_toolbar=True
            )
        
        # Procesar líneas dibujadas
        lines = []
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            
            for idx, obj in enumerate(objects):
                if obj["type"] == "line":
                    x1, y1 = int(obj["x1"]), int(obj["y1"])
                    x2, y2 = int(obj["x2"]), int(obj["y2"])
                    
                    lines.append({
                        'name': f"{line_name}_{idx}" if len(objects) > 1 else line_name,
                        'direction': direction,
                        'coords': [(x1, y1), (x2, y2)],
                        'color': line_color,
                        'stroke_width': stroke_width
                    })
        
        # Agregar líneas existentes
        if existing_lines:
            lines.extend(existing_lines)
        
        return lines
    
    def draw_polygons(self, existing_zones=None):
        """
        Interfaz para dibujar zonas poligonales.
        
        Args:
            existing_zones: Lista de zonas existentes para mostrar
            
        Returns:
            Lista de zonas dibujadas con formato:
            [{'name': str, 'type': str, 'coords': [(x1,y1), (x2,y2), ...], 'color': str}]
        """
        st.subheader("🔷 Dibujar Zonas Poligonales")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("**Configuración de Zona**")
            
            zone_name = st.text_input(
                "Nombre de la Zona",
                value=f"Zona_{len(existing_zones) + 1 if existing_zones else 1}",
                key=f"zone_name_{self.canvas_key}"
            )
            
            zone_type = st.selectbox(
                "Tipo de Zona",
                ["Conteo", "Exclusión", "Giro Prohibido", "Presencia", "Zona de Calor"],
                key=f"zone_type_{self.canvas_key}"
            )
            
            zone_color = st.color_picker(
                "Color de la Zona",
                "#FF0000",
                key=f"zone_color_{self.canvas_key}"
            )
            
            stroke_width = st.slider(
                "Grosor del Borde",
                min_value=2,
                max_value=10,
                value=3,
                key=f"zone_stroke_{self.canvas_key}"
            )
            
            drawing_mode = st.radio(
                "Modo de Dibujo",
                ["Rectángulo", "Polígono Libre"],
                key=f"drawing_mode_{self.canvas_key}"
            )
            
            st.info("💡 **Instrucciones:**\n\n"
                   "**Rectángulo:**\n"
                   "- Arrastra para crear un rectángulo\n\n"
                   "**Polígono:**\n"
                   "- Haz clic para cada vértice\n"
                   "- Doble clic para cerrar")
        
        with col1:
            # Determinar modo de dibujo
            mode = "rect" if drawing_mode == "Rectángulo" else "polygon"
            
            # Canvas para dibujar
            canvas_result = st_canvas(
                fill_color=f"{zone_color}33",  # Color con transparencia
                stroke_width=stroke_width,
                stroke_color=zone_color,
                background_image=self.background_image,
                update_streamlit=True,
                height=self.height,
                width=self.width,
                drawing_mode=mode,
                point_display_radius=3 if mode == "polygon" else 0,
                key=self.canvas_key,
                display_toolbar=True
            )
        
        # Procesar zonas dibujadas
        zones = []
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            
            for idx, obj in enumerate(objects):
                coords = []
                
                if obj["type"] == "rect":
                    # Convertir rectángulo a polígono
                    x, y = int(obj["left"]), int(obj["top"])
                    w, h = int(obj["width"]), int(obj["height"])
                    coords = [
                        (x, y),
                        (x + w, y),
                        (x + w, y + h),
                        (x, y + h)
                    ]
                
                elif obj["type"] == "path":
                    # Polígono libre
                    path = obj.get("path", [])
                    for point in path:
                        if len(point) >= 3:  # ["L", x, y] o ["M", x, y]
                            coords.append((int(point[1]), int(point[2])))
                
                if coords:
                    zones.append({
                        'name': f"{zone_name}_{idx}" if len(objects) > 1 else zone_name,
                        'type': zone_type,
                        'coords': coords,
                        'color': zone_color,
                        'stroke_width': stroke_width
                    })
        
        # Agregar zonas existentes
        if existing_zones:
            zones.extend(existing_zones)
        
        return zones
    
    @staticmethod
    def visualize_annotations(frame, lines=None, zones=None):
        """
        Dibuja las líneas y zonas sobre el frame.
        
        Args:
            frame: Frame de video (numpy array BGR)
            lines: Lista de líneas
            zones: Lista de zonas
            
        Returns:
            Frame con anotaciones dibujadas
        """
        annotated_frame = frame.copy()
        
        # Dibujar zonas
        if zones:
            for zone in zones:
                pts = np.array(zone['coords'], np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Color BGR (convertir desde hex si es necesario)
                color = DrawingTool._hex_to_bgr(zone.get('color', '#FF0000'))
                
                # Dibujar polígono relleno con transparencia
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                
                # Dibujar borde
                cv2.polylines(annotated_frame, [pts], True, color, 
                             zone.get('stroke_width', 3))
                
                # Etiqueta
                cv2.putText(annotated_frame, zone['name'], 
                           zone['coords'][0], 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
        
        # Dibujar líneas
        if lines:
            for line in lines:
                color = DrawingTool._hex_to_bgr(line.get('color', '#00FF00'))
                
                cv2.line(annotated_frame, 
                        line['coords'][0], 
                        line['coords'][1], 
                        color, 
                        line.get('stroke_width', 4))
                
                # Etiqueta con dirección
                mid_point = (
                    (line['coords'][0][0] + line['coords'][1][0]) // 2,
                    (line['coords'][0][1] + line['coords'][1][1]) // 2
                )
                
                label = f"{line['name']} ({line['direction']})"
                cv2.putText(annotated_frame, label, mid_point, 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
                
                # Flecha para indicar dirección
                DrawingTool._draw_arrow(annotated_frame, 
                                       line['coords'][0], 
                                       line['coords'][1], 
                                       color)
        
        return annotated_frame
    
    @staticmethod
    def _hex_to_bgr(hex_color):
        """Convierte color hexadecimal a BGR."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])  # RGB a BGR
    
    @staticmethod
    def _draw_arrow(img, start, end, color, thickness=2):
        """Dibuja una flecha para indicar dirección."""
        cv2.arrowedLine(img, start, end, color, thickness, tipLength=0.3)
    
    @staticmethod
    def export_config(lines, zones, filename):
        """Exporta la configuración a JSON."""
        config = {
            'lines': lines,
            'zones': zones,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        
        return filename
    
    @staticmethod
    def import_config(filename):
        """Importa configuración desde JSON."""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        return config.get('lines', []), config.get('zones', [])