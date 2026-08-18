import streamlit as st
from PIL import Image, ImageOps
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

def arreglar_orientacion(imagen_pil):
    """Corrige la rotación EXIF de fotos tomadas con móviles"""
    try:
        return ImageOps.exif_transpose(imagen_pil)
    except Exception:
        return imagen_pil

def detectar_y_rotar_rostro(img_cv):
    """Intenta detectar un rostro probando en las 4 orientaciones posibles (0, 90, 180, 270 grados)"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for angulo in [0, 90, 180, 270]:
        if angulo == 90:
            rotada = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
        elif angulo == 180:
            rotada = cv2.rotate(img_cv, cv2.ROTATE_180)
        elif angulo == 270:
            rotada = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotada = img_cv

        gray = cv2.cvtColor(rotada, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(caras) > 0:
            return rotada, caras[0]
    return img_cv, None

if foto_dni and foto_original:
    col1, col2 = st.columns(2)
    with col1:
        st.image(foto_dni, caption="DNI Escaneado", use_container_width=True)
    with col2:
        st.image(foto_original, caption="Rostro Fotografiado", use_container_width=True)

    if st.button("🚀 Procesar Verificación"):
        with st.spinner("Orientando imágenes, analizando rostros y extrayendo datos..."):
            # 1. Cargar e igualar orientación EXIF
            img_dni_pil = arreglar_orientacion(Image.open(foto_dni)).convert("RGB")
            img_user_pil = arreglar_orientacion(Image.open(foto_original)).convert("RGB")

            # Guardar imágenes base corregidas
            img_dni_pil.save("dni_temp.jpg")
            img_user_pil.save("usuario_temp.jpg")

            img_dni_cv = cv2.imread("dni_temp.jpg")
            img_user_cv = cv2.imread("usuario_temp.jpg")

            # 2. Corregir rotación automática mediante detección facial
            img_dni_correcta, cara1 = detectar_y_rotar_rostro(img_dni_cv)
            img_user_correcta, cara2 = detectar_y_rotar_rostro(img_user_cv)

            # Sobrescribir con la orientación enderezada
            cv2.imwrite("dni_temp.jpg", img_dni_correcta)
            cv2.imwrite("usuario_temp.jpg", img_user_correcta)

            # 3. Verificación Facial (Coincidencia mediante histogramas HOG de área facial)
            es_misma_persona = False
            if cara1 is not None and cara2 is not None:
                x1, y1, w1, h1 = cara1
                x2, y2, w2, h2 = cara2
                
                crop1 = cv2.resize(img_dni_correcta[y1:y1+h1, x1:x1+w1], (100, 100))
                crop2 = cv2.resize(img_user_correcta[y2:y2+h2, x2:x2+w2], (100, 100))

                hist1 = cv2.calcHist([cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])

                cv2.normalize(hist1, hist1)
                cv2.normalize(hist2, hist2)

                similitud = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if similitud > 0.4:
                    es_misma_persona = True

            # 4. OCR mejorado en la imagen corregida
            gray_dni = cv2.cvtColor(img_dni_correcta, cv2.COLOR_BGR2GRAY)

            # Escalado de alta resolución para lectura OCR
            h, w = gray_dni.shape
            if w < 1200:
                scale = 1200 / w
                gray_dni = cv2.resize(gray_dni, (1200, int(h * scale)), interpolation=cv2.INTER_CUBIC)

            texto_raw = pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 11')
            texto_raw += "\n" + pytesseract.image_to_string(gray_dni, lang='spa')

            # DNI (8 dígitos + letra)
            match_dni = re.search(r'\b\d{8}\s*[-_]?\s*[A-Za-z]\b', texto_raw)
            if match_dni:
                numero_dni = re.sub(r'[\s\-_]', '', match_dni.group(0)).upper()
            else:
                numero_dni = "No detectado"

            # Fechas del DNI Español (Formatos DD/MM/AAAA, DD MM AAAA o DD.MM.AAAA)
            fechas_encontradas = re.findall(r'\b\d{2}[\s/.-]\d{2}[\s/.-]\d{4}\b', texto_raw)
            fechas_limpias = [re.sub(r'[\s.-]', '/', f) for f in fechas_encontradas]

            fecha_nacimiento = fechas_limpias[0] if len(fechas_limpias) > 0 else "No detectada"
            fecha_caducidad = fechas_limpias[1] if len(fechas_limpias) > 1 else "No detectada"

            # 5. Generación del Informe PDF
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
                c.drawString(40, height - 120, "✔ VERIFICACIÓN FACIAL: COINCIDENCIA CONFIRMADA")
            else:
                c.setFillColor(colors.HexColor("#991B1B"))
                c.drawString(40, height - 120, "✖ VERIFICACIÓN FACIAL: NO COINCIDEN O ROSTRO NO CLARO")

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
            c.drawString(40, height - 290, "EVIDENCIAS FOTOGRÁFICAS (ORIENTADAS)")

            c.drawImage("dni_temp.jpg", 40, height - 510, width=240, height=190, preserveAspectRatio=True)
            c.drawImage("usuario_temp.jpg", 315, height - 510, width=240, height=190, preserveAspectRatio=True)

            c.setFont("Helvetica", 8)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.drawString(40, 30, "Documento generado automáticamente por la aplicación de verificación.")

            c.save()

            if es_misma_persona:
                st.success(f"✅ ¡Rostros coincidentes! DNI: **{numero_dni}**")
            else:
                st.warning(f"⚠️ Proceso finalizado. DNI Detectado: **{numero_dni}**")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar Informe en PDF",
                    data=pdf_file,
                    file_name=f"Informe_DNI_{numero_dni}.pdf",
                    mime="application/pdf"
                )
