import streamlit as st
from PIL import Image
import pytesseract
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from deepface import DeepFace
import re

st.set_page_config(page_title="Verificación DNI Pro", layout="centered")
st.title("🆔 Verificación de DNI y Comparativa Facial")

foto_dni = st.file_uploader("1. Sube la foto del DNI", type=["jpg", "png", "jpeg"])
foto_original = st.file_uploader("2. Sube la foto del usuario (Selfie)", type=["jpg", "png", "jpeg"])

if foto_dni and foto_original:
    col1, col2 = st.columns(2)
    with col1:
        st.image(foto_dni, caption="DNI Escaneado", use_container_width=True)
    with col2:
        st.image(foto_original, caption="Rostro Fotografiado", use_container_width=True)

    if st.button("🚀 Procesar Verificación"):
        with st.spinner("Analizando información y generando informe..."):
            # Guardar archivos temporales
            with open("dni_temp.jpg", "wb") as f:
                f.write(foto_dni.getbuffer())
            with open("usuario_temp.jpg", "wb") as f:
                f.write(foto_original.getbuffer())

            # 1. Comparación Facial
            try:
                resultado = DeepFace.verify(img1_path="dni_temp.jpg", img2_path="usuario_temp.jpg", enforce_detection=False)
                es_misma_persona = resultado.get("verified", False)
            except Exception:
                es_misma_persona = False

            # 2. Extracción OCR y Filtrado con Expresiones Regulares
            imagen_dni = Image.open("dni_temp.jpg")
            texto_raw = pytesseract.image_to_string(imagen_dni, lang='spa')

            # Buscar patrón de DNI (8 dígitos + letra)
            match_dni = re.search(r'\b\d{8}[A-Za-z]\b', texto_raw)
            numero_dni = match_dni.group(0).upper() if match_dni else "No detectado"

            # Buscar patrones de fechas (DD/MM/AAAA o DD-MM-AAAA)
            fechas_encontradas = re.findall(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', texto_raw)
            fecha_nacimiento = fechas_encontradas[0] if len(fechas_encontradas) > 0 else "No detectada"
            fecha_caducidad = fechas_encontradas[1] if len(fechas_encontradas) > 1 else "No detectada"

            # 3. Diseño Estructurado del PDF (ReportLab)
            pdf_path = "informe_verificacion.pdf"
            c = canvas.Canvas(pdf_path, pagesize=A4)
            width, height = A4

            # --- ENCABEZADO Y LOGO/BANNER ---
            c.setFillColor(colors.HexColor("#1E3A8A")) # Azul oscuro institucional
            c.rect(0, height - 90, width, 90, fill=True, stroke=False)
            
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(40, height - 40, "MI EMPRESA / APLICACIÓN")
            c.setFont("Helvetica", 11)
            c.drawString(40, height - 60, "Informe Oficial de Verificación de Identidad")

            # --- ESTADO DE COMPARACIÓN FACIAL ---
            c.setFont("Helvetica-Bold", 13)
            if es_misma_persona:
                c.setFillColor(colors.HexColor("#166534")) # Verde
                c.drawString(40, height - 120, "✔ VERIFICACIÓN FACIAL: COINCIDENCIA CONFIRMADA")
            else:
                c.setFillColor(colors.HexColor("#991B1B")) # Rojo
                c.drawString(40, height - 120, "✖ VERIFICACIÓN FACIAL: NO COINCIDEN LOS ROSTROS")

            # --- TABLA DE DATOS FILTRADOS ---
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 160, "DATOS EXTRAÍDOS DEL DOCUMENTO")

            # Dibujar caja de datos
            c.setStrokeColor(colors.HexColor("#E2E8F0"))
            c.setFillColor(colors.HexColor("#F8FAFC"))
            c.roundRect(40, height - 260, width - 80, 85, 6, fill=True, stroke=True)

            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(60, height - 195, "Número de DNI / NIE:")
            c.drawString(60, height - 215, "Fecha de Nacimiento:")
            c.drawString(60, height - 235, "Fecha de Caducidad:")

            c.setFont("Helvetica", 10)
            c.setFillColor(colors.HexColor("#0F172A"))
            c.drawString(220, height - 195, numero_dni)
            c.drawString(220, height - 215, fecha_nacimiento)
            c.drawString(220, height - 235, fecha_caducidad)

            # --- FOTOGRAFÍAS ADJUNTAS ---
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 290, "EVIDENCIAS FOTOGRÁFICAS")

            # Mostrar imágenes en paralelo
            c.drawImage("dni_temp.jpg", 40, height - 510, width=240, height=190, preserveAspectRatio=True)
            c.drawImage("usuario_temp.jpg", 315, height - 510, width=240, height=190, preserveAspectRatio=True)

            # --- PIE DE PÁGINA ---
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.drawString(40, 30, "Este documento ha sido generado de forma automática mediante la aplicación de verificación de DNI.")

            c.save()

            # Confirmación en pantalla de Streamlit
            if es_misma_persona:
                st.success(f"✅ ¡Rostros coincidentes! DNI Detectado: **{numero_dni}**")
            else:
                st.warning(f"⚠️ Los rostros no coinciden. DNI Detectado: **{numero_dni}**")

            # Botón de Descarga
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar Informe Personalizado en PDF",
                    data=pdf_file,
                    file_name=f"Informe_DNI_{numero_dni}.pdf",
                    mime="application/pdf"
                )
