import streamlit as st
from PIL import Image, ImageOps
import pytesseract
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import cv2
import numpy as np
import re
import urllib.request
import os

st.set_page_config(page_title="Verificación DNI Pro", layout="centered")
st.title("🆔 Verificación de DNI y Comparativa Facial")

# Descargar detector de rostros de respaldo si no existe en el servidor
CASCADE_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, CASCADE_PATH)

def rotar_imagen(pil_img, grados):
    """Rota una imagen PIL según el ángulo seleccionado por el usuario"""
    if grados == 90:
        return pil_img.rotate(-90, expand=True)
    elif grados == 180:
        return pil_img.rotate(180, expand=True)
    elif grados == 270:
        return pil_img.rotate(-270, expand=True)
    return pil_img

foto_dni = st.file_uploader("1. Sube la foto del DNI", type=["jpg", "png", "jpeg"])
foto_original = st.file_uploader("2. Sube la foto del usuario (Selfie)", type=["jpg", "png", "jpeg"])

if foto_dni and foto_original:
    # Cargar imágenes arreglando orientación básica EXIF
    img_dni_pil = ImageOps.exif_transpose(Image.open(foto_dni)).convert("RGB")
    img_user_pil = ImageOps.exif_transpose(Image.open(foto_original)).convert("RGB")

    st.markdown("---")
    st.subheader("🔄 Ajuste de Rotación (Asegúrate de que ambas fotos se vean derechas)")

    col_rot1, col_rot2 = st.columns(2)
    with col_rot1:
        rot_dni = st.selectbox("Girar foto DNI:", [0, 90, 180, 270], format_func=lambda x: f"{x}°", key="rot_dni")
        img_dni_pil = rotar_imagen(img_dni_pil, rot_dni)
        st.image(img_dni_pil, caption="DNI Orientado", use_container_width=True)

    with col_rot2:
        rot_user = st.selectbox("Girar foto Selfie:", [0, 90, 180, 270], format_func=lambda x: f"{x}°", key="rot_user")
        img_user_pil = rotar_imagen(img_user_pil, rot_user)
        st.image(img_user_pil, caption="Rostro Orientado", use_container_width=True)

    if st.button("🚀 Procesar Verificación"):
        with st.spinner("Analizando información y generando informe..."):
            # Guardar fotos orientadas
            img_dni_pil.save("dni_temp.jpg")
            img_user_pil.save("usuario_temp.jpg")

            img1_cv = cv2.imread("dni_temp.jpg")
            img2_cv = cv2.imread("usuario_temp.jpg")

            # 1. Detección Facial con OpenCV usando el archivo local
            face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
            gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)

            faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=3)
            faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=3)

            es_misma_persona = False
            if len(faces1) > 0 and len(faces2) > 0:
                x1, y1, w1, h1 = faces1[0]
                x2, y2, w2, h2 = faces2[0]

                crop1 = cv2.resize(gray1[y1:y1+h1, x1:x1+w1], (100, 100))
                crop2 = cv2.resize(gray2[y2:y2+h2, x2:x2+w2], (100, 100))

                hist1 = cv2.calcHist([crop1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([crop2], [0], None, [256], [0, 256])

                cv2.normalize(hist1, hist1)
                cv2.normalize(hist2, hist2)

                similitud = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if similitud > 0.35:
                    es_misma_persona = True

            # 2. OCR en DNI
            h, w = gray1.shape
            if w < 1200:
                scale = 1200 / w
                gray1 = cv2.resize(gray1, (1200, int(h * scale)), interpolation=cv2.INTER_CUBIC)

            texto_raw = pytesseract.image_to_string(gray1, lang='spa', config='--psm 11')
            texto_raw += "\n" + pytesseract.image_to_string(gray1, lang='spa')

            # Extraer DNI
            match_dni = re.search(r'\b\d{8}\s*[-_]?\s*[A-Za-z]\b', texto_raw)
            if match_dni:
                numero_dni = re.sub(r'[\s\-_]', '', match_dni.group(0)).upper()
            else:
                numero_dni = "No detectado"

            # Extraer Fechas DNI (formatos DD MM AAAA, DD/MM/AAAA, DD-MM-AAAA)
            fechas_encontradas = re.findall(r'\b\d{2}[\s/.-]\d{2}[\s/.-]\d{4}\b', texto_raw)
            fechas_limpias = [re.sub(r'[\s.-]', '/', f) for f in fechas_encontradas]

            fecha_nacimiento = fechas_limpias[0] if len(fechas_limpias) > 0 else "No detectada"
            fecha_caducidad = fechas_limpias[1] if len(fechas_limpias) > 1 else "No detectada"

            # 3. PDF
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
            c.drawString(40, height - 290, "EVIDENCIAS FOTOGRÁFICAS")

            c.drawImage("dni_temp.jpg", 40, height - 510, width=240, height=190, preserveAspectRatio=True)
            c.drawImage("usuario_temp.jpg", 315, height - 510, width=240, height=190, preserveAspectRatio=True)

            c.setFont("Helvetica", 8)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.drawString(40, 30, "Documento generado automáticamente por la aplicación de verificación.")

            c.save()

            if es_misma_persona:
                st.success(f"✅ ¡Rostros coincidentes! DNI Detectado: **{numero_dni}**")
            else:
                st.warning(f"⚠️ Proceso completado. DNI Detectado: **{numero_dni}**")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar Informe en PDF",
                    data=pdf_file,
                    file_name=f"Informe_DNI_{numero_dni}.pdf",
                    mime="application/pdf"
                )
