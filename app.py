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

CASCADE_PATH = "haarcascade_frontalface_default.xml"

def obtener_cascade():
    """Descarga de forma segura el clasificador de rostros Haar Cascade si no existe"""
    if not os.path.exists(CASCADE_PATH) or os.path.getsize(CASCADE_PATH) < 1000:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(CASCADE_PATH, 'wb') as out_file:
            out_file.write(response.read())

def detectar_y_recortar_rostro(img_bgr):
    """Detecta la cara en la imagen y devuelve el recorte en escala de grises"""
    try:
        obtener_cascade()
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        gray_eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            return gray[y:y+h, x:x+w]
        return None
    except Exception:
        return None

def comparar_rostros(img1_bgr, img2_bgr):
    """Compara los recortes de rostro usando extracción de puntos clave (ORB) y coincidencia"""
    face1 = detectar_y_recortar_rostro(img1_bgr)
    face2 = detectar_y_recortar_rostro(img2_bgr)

    if face1 is None or face2 is None:
        gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
        face1 = cv2.resize(gray1, (150, 150))
        face2 = cv2.resize(gray2, (150, 150))
    else:
        face1 = cv2.resize(face1, (150, 150))
        face2 = cv2.resize(face2, (150, 150))

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(face1, None)
    kp2, des2 = orb.detectAndCompute(face2, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    buenas = [m for m in matches if m.distance < 60]

    return len(buenas) >= 8

def rotar_imagen(pil_img, grados):
    """Rota la imagen PIL según la elección del usuario"""
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
    img_dni_pil = ImageOps.exif_transpose(Image.open(foto_dni)).convert("RGB")
    img_user_pil = ImageOps.exif_transpose(Image.open(foto_original)).convert("RGB")

    st.markdown("---")
    st.subheader("🔄 Ajuste de Rotación")

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
        with st.spinner("Procesando fotos, analizando rostro y extrayendo texto..."):
            path_dni = "dni_temp.jpg"
            path_user = "usuario_temp.jpg"
            img_dni_pil.save(path_dni)
            img_user_pil.save(path_user)

            img1_cv = cv2.imread(path_dni)
            img2_cv = cv2.imread(path_user)

            # 1. Comparación de rostro
            es_misma_persona = comparar_rostros(img1_cv, img2_cv)

            # 2. OCR
            gray_dni = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
            h, w = gray_dni.shape
            if w < 1400:
                scale = 1400 / w
                gray_dni = cv2.resize(gray_dni, (1400, int(h * scale)), interpolation=cv2.INTER_CUBIC)

            texto_raw = pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 11')
            texto_raw += "\n" + pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 6')

            # Extraer DNI
            match_dni = re.search(r'\b\d{8}\s*[-_]?\s*[A-Za-z]\b', texto_raw)
            numero_dni = re.sub(r'[\s\-_]', '', match_dni.group(0)).upper() if match_dni else "No detectado"

            # Extraer Fechas
            patron_fechas = r'\b(\d{2})[\s/.-](\d{2})[\s/.-](\d{4})\b'
            fechas_coincidentes = re.findall(patron_fechas, texto_raw)

            fechas_formateadas = [f"{d}/{m}/{a}" for d, m, a in fechas_coincidentes]

            fecha_nacimiento = "No detectada"
            fecha_caducidad = "No detectada"

            if len(fechas_formateadas) >= 3:
                fecha_nacimiento = fechas_formateadas[0]
                fecha_caducidad = fechas_formateadas[2]
            elif len(fechas_formateadas) == 2:
                fecha_nacimiento = fechas_formateadas[0]
                fecha_caducidad = fechas_formateadas[1]
            elif len(fechas_formateadas) == 1:
                fecha_nacimiento = fechas_formateadas[0]

            match_validez = re.search(r'(?:VALIDEZ|VAL)[^\d]*(\d{2}[\s/.-]\d{2}[\s/.-]\d{4})', texto_raw, re.IGNORECASE)
            if match_validez:
                f_val = match_validez.group(1).replace(' ', '/').replace('.', '/').replace('-', '/')
                fecha_caducidad = f_val

            # 3. Generar PDF
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
            # Subtítulo modificado
            c.drawString(40, height - 60, "Informe de Verificación de Identidad")

            # Resultado Verificación
            c.setFont("Helvetica-Bold", 13)
            if es_misma_persona:
                c.setFillColor(colors.HexColor("#166534"))
                c.drawString(40, height - 120, "✔ VERIFICACIÓN FACIAL: COINCIDENCIA CONFIRMADA")
            else:
                c.setFillColor(colors.HexColor("#991B1B"))
                c.drawString(40, height - 120, "✖ VERIFICACIÓN FACIAL: REVISIÓN MANUAL REQUERIDA")

            # Datos Extraídos
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

            # Evidencias
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 290, "EVIDENCIAS FOTOGRÁFICAS")

            c.drawImage(path_dni, 40, height - 510, width=240, height=190, preserveAspectRatio=True)
            c.drawImage(path_user, 315, height - 510, width=240, height=190, preserveAspectRatio=True)

            # Nota a pie de página modificada
            c.setFont("Helvetica-Oblique", 7.5)
            c.setFillColor(colors.HexColor("#64748B"))
            c.drawString(40, 30, "Aviso: Este documento no es oficial y se basa únicamente en un reconocimiento facial de características biométricas.")

            c.save()

            if es_misma_persona:
                st.success(f"✅ ¡VERIFICACIÓN FACIAL CONFIRMADA! DNI: **{numero_dni}**")
            else:
                st.warning(f"⚠️ Proceso completado. DNI: **{numero_dni}**")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar Informe en PDF",
                    data=pdf_file,
                    file_name=f"Informe_DNI_{numero_dni}.pdf",
                    mime="application/pdf"
                )
