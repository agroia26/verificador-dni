import streamlit as st
from PIL import Image
import pytesseract
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import cv2
import numpy as np
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
            # Guardar imágenes temporalmente
            with open("dni_temp.jpg", "wb") as f:
                f.write(foto_dni.getbuffer())
            with open("usuario_temp.jpg", "wb") as f:
                f.write(foto_original.getbuffer())

            # 1. Detección facial rápida con OpenCV
            es_misma_persona = False
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                img1 = cv2.imread("dni_temp.jpg", cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread("usuario_temp.jpg", cv2.IMREAD_GRAYSCALE)

                faces1 = face_cascade.detectMultiScale(img1, 1.1, 4)
                faces2 = face_cascade.detectMultiScale(img2, 1.1, 4)

                if len(faces1) > 0 and len(faces2) > 0:
                    es_misma_persona = True
            except Exception:
                es_misma_persona = False

            # 2. Preprocesamiento de la foto para mejorar la lectura OCR
            img_cv = cv2.imread("dni_temp.jpg")
            
            # Agrandar la imagen para darle nitidez a los caracteres pequeños
            h, w = img_cv.shape[:2]
            if w < 1200:
                scale = 1200 / w
                img_cv = cv2.resize(img_cv, (1200, int(h * scale)), interpolation=cv2.INTER_CUBIC)

            # Convertir a escala de grises y subir contraste
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            gray = cv2.addWeighted(gray, 1.5, gray, -0.5, 0)

            # Lectura combinada (versión limpia + foto original)
            texto_preprocesado = pytesseract.image_to_string(gray, lang='spa', config='--psm 11')
            texto_original = pytesseract.image_to_string(Image.open("dni_temp.jpg"), lang='spa')
            texto_total = texto_preprocesado + "\n" + texto_original

            # Búsqueda flexible de DNI (tolera espacios o guiones entre números y letra)
            match_dni = re.search(r'\b\d{8}\s*[-_]?\s*[A-Za-z]\b', texto_total)
            if match_dni:
                numero_dni = re.sub(r'[\s\-_]', '', match_dni.group(0)).upper()
            else:
                numero_dni = "No detectado"

            # Búsqueda flexible de Fechas (DD/MM/AAAA, DD-MM-AAAA o DD.MM.AAAA)
            fechas_encontradas = re.findall(r'\b\d{2}[/.-]\d{2}[/.-]\d{4}\b', texto_total)
            fecha_nacimiento = fechas_encontradas[0].replace('.', '/') if len(fechas_encontradas) > 0 else "No detectada"
            fecha_caducidad = fechas_encontradas[1].replace('.', '/') if len(fechas_encontradas) > 1 else "No detectada"

            # 3. Generación del informe PDF
            pdf_path = "informe_verificacion.pdf"
            c = canvas.Canvas(pdf_path, pagesize=A4)
            width, height = A4

            # Encabezado
            c.setFillColor(colors.HexColor("#1E3A8A"))
            c.rect(0, height - 90, width, 90, fill=True, stroke=False)
            
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(40, height - 40, "VERIFICACIÓN DE IDENTIDAD")
            c.setFont("Helvetica", 11)
            c.drawString(40, height - 60, "Informe Oficial de Verificación de Identidad")

            # Estado
            c.setFont("Helvetica-Bold", 13)
            if es_misma_persona:
                c.setFillColor(colors.HexColor("#166534"))
                c.drawString(40, height - 120, "✔ VERIFICACIÓN FACIAL: ROSTROS DETECTADOS Y PROCESADOS")
            else:
                c.setFillColor(colors.HexColor("#991B1B"))
                c.drawString(40, height - 120, "✖ VERIFICACIÓN FACIAL: NO SE DETECTÓ ROSTRO CLARO")

            # Datos
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 160, "DATOS EXTRAÍDOS DEL DOCUMENTO")

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

            # Fotos
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 290, "EVIDENCIAS FOTOGRÁFICAS")

            c.drawImage("dni_temp.jpg", 40, height - 510, width=240, height=190, preserveAspectRatio=True)
            c.drawImage("usuario_temp.jpg", 315, height - 510, width=240, height=190, preserveAspectRatio=True)

            # Pie de página
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.drawString(40, 30, "Documento generado automáticamente por la aplicación de verificación.")

            c.save()

            if numero_dni != "No detectado":
                st.success(f"✅ ¡DNI Detectado con éxito!: **{numero_dni}**")
            else:
                st.warning("⚠️ No se pudo leer el número de DNI. Asegúrate de que la foto esté bien enfocada y sin reflejos.")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar Informe en PDF",
                    data=pdf_file,
                    file_name=f"Informe_DNI_{numero_dni}.pdf",
                    mime="application/pdf"
                )
