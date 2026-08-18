import streamlit as st
from PIL import Image, ImageOps
import pytesseract
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import cv2
import numpy as np
import re
import os

# Intentar importar DeepFace para comparación facial mediante IA
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except Exception:
    HAS_DEEPFACE = False

st.set_page_config(page_title="Verificación DNI Pro", layout="centered")
st.title("🆔 Verificación de DNI y Comparativa Facial")

def rotar_imagen(pil_img, grados):
    """Rota una imagen PIL según el ángulo seleccionado"""
    if grados == 90:
        return pil_img.rotate(-90, expand=True)
    elif grados == 180:
        return pil_img.rotate(180, expand=True)
    elif grados == 270:
        return pil_img.rotate(-270, expand=True)
    return pil_img

def verificar_similitud_facial(img1_path, img2_path):
    """Compara los rostros utilizando IA (DeepFace) o coincidencia de puntos clave de OpenCV"""
    if HAS_DEEPFACE:
        try:
            # Modelo VGG-Face / Facenet para calcular distancia entre rasgos
            res = DeepFace.verify(
                img1_path=img1_path, 
                img2_path=img2_path, 
                model_name="VGG-Face", 
                enforce_detection=False
            )
            return res.get("verified", False), float(res.get("distance", 1.0))
        except Exception:
            pass

    # Fallback avanzado con OpenCV (ORB Feature Matching) si DeepFace aún carga
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return False, 1.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)
        buenas_coincidencias = [m for m in matches if m.distance < 50]

        if len(buenas_coincidencias) > 15:
            return True, 0.3
        return False, 0.8
    except Exception:
        return False, 1.0

foto_dni = st.file_uploader("1. Sube la foto del DNI", type=["jpg", "png", "jpeg"])
foto_original = st.file_uploader("2. Sube la foto del usuario (Selfie)", type=["jpg", "png", "jpeg"])

if foto_dni and foto_original:
    img_dni_pil = ImageOps.exif_transpose(Image.open(foto_dni)).convert("RGB")
    img_user_pil = ImageOps.exif_transpose(Image.open(foto_original)).convert("RGB")

    st.markdown("---")
    st.subheader("🔄 Ajuste de Rotación (Asegúrate de que ambas fotos estén verticales)")

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
        with st.spinner("Analizando redes neuronales faciales y extrayendo OCR..."):
            # Guardar archivos
            path_dni = "dni_temp.jpg"
            path_user = "usuario_temp.jpg"
            img_dni_pil.save(path_dni)
            img_user_pil.save(path_user)

            # 1. Comparación Facial con IA
            es_misma_persona, distancia = verificar_similitud_facial(path_dni, path_user)

            # 2. OCR y Extracción de Datos
            img1_cv = cv2.imread(path_dni)
            gray_dni = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)

            h, w = gray_dni.shape
            if w < 1400:
                scale = 1400 / w
                gray_dni = cv2.resize(gray_dni, (1400, int(h * scale)), interpolation=cv2.INTER_CUBIC)

            gray_dni = cv2.addWeighted(gray_dni, 1.5, gray_dni, -0.5, 0)

            texto_raw = pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 11')
            texto_raw += "\n" + pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 6')

            # DNI
            match_dni = re.search(r'\b\d{8}\s*[-_]?\s*[A-Za-z]\b', texto_raw)
            numero_dni = re.sub(r'[\s\-_]', '', match_dni.group(0)).upper() if match_dni else "No detectado"

            # Fechas
            patron_fechas = r'\b(\d{2})[\s/.-](\d{2})[\s/.-](\d{4}|\d{2})\b'
            fechas_coincidentes = re.findall(patron_fechas, texto_raw)

            fechas_formateadas = []
            for f in fechas_coincidentes:
                dia, mes, anio = f
                if len(anio) == 2:
                    anio = "20" + anio if int(anio) < 50 else "19" + anio
                fechas_formateadas.append(f"{dia}/{mes}/{anio}")

            fecha_nacimiento = fechas_formateadas[0] if len(fechas_formateadas) > 0 else "No detectada"
            fecha_caducidad = fechas_formateadas[1] if len(fechas_formateadas) > 1 else "No detectada"

            # 3. Generación de PDF
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
                c.drawString(40, height - 120, "✖ VERIFICACIÓN FACIAL: NO COINCIDEN / REVISIÓN REQUERIDA")

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

            c.setFont("Helvetica", 8)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.drawString(40, 30, "Documento generado automáticamente por la aplicación de verificación.")

            c.save()

            if es_misma_persona:
                st.success(f"✅ ¡VERIFICACIÓN FACIAL CORRECTA! Coincidencia confirmada para el DNI: **{numero_dni}**")
            else:
                st.warning(f"⚠️ DNI detectado: **{numero_dni}**, pero la prueba de rostro requiere revisión manual.")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar Informe en PDF",
                    data=pdf_file,
                    file_name=f"Informe_DNI_{numero_dni}.pdf",
                    mime="application/pdf"
                )
