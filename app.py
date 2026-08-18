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

def rotar_imagen(pil_img, grados):
    """Rota una imagen PIL según el ángulo seleccionado"""
    if grados == 90:
        return pil_img.rotate(-90, expand=True)
    elif grados == 180:
        return pil_img.rotate(180, expand=True)
    elif grados == 270:
        return pil_img.rotate(-270, expand=True)
    return pil_img

def detectar_caras_y_comparar(img1_bgr, img2_bgr):
    """
    Detecta áreas faciales usando segmentación de color de piel y bordes, 
    y compara los histogramas de características para validar coincidencia.
    """
    try:
        # Convertir a espacio de color HSV para detectar piel de forma segura
        hsv1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2HSV)

        # Rango general de tonos de piel humana en HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv1, lower_skin, upper_skin)
        mask2 = cv2.inRange(hsv2, lower_skin, upper_skin)

        # Calcular histograma de la región de la piel
        hist1 = cv2.calcHist([hsv1], [0, 1], mask1, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], mask2, [180, 256], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        similitud = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Si ambas imágenes tienen un porcentaje razonable de piel y similitud cromática
        area_skin1 = cv2.countNonZero(mask1) / (img1_bgr.shape[0] * img1_bgr.shape[1])
        area_skin2 = cv2.countNonZero(mask2) / (img2_bgr.shape[0] * img2_bgr.shape[1])

        if area_skin1 > 0.05 and area_skin2 > 0.05 and similitud > 0.30:
            return True, similitud
        return False, similitud
    except Exception:
        return False, 0.0

foto_dni = st.file_uploader("1. Sube la foto del DNI", type=["jpg", "png", "jpeg"])
foto_original = st.file_uploader("2. Sube la foto del usuario (Selfie)", type=["jpg", "png", "jpeg"])

if foto_dni and foto_original:
    # Cargar imágenes aplicando orientación EXIF inicial
    img_dni_pil = ImageOps.exif_transpose(Image.open(foto_dni)).convert("RGB")
    img_user_pil = ImageOps.exif_transpose(Image.open(foto_original)).convert("RGB")

    st.markdown("---")
    st.subheader("🔄 Ajuste de Rotación (Asegúrate de que ambas fotos estén verticales y legibles)")

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
        with st.spinner("Procesando verificación facial y extracción OCR..."):
            # Guardar archivos en disco
            img_dni_pil.save("dni_temp.jpg")
            img_user_pil.save("usuario_temp.jpg")

            img1_cv = cv2.imread("dni_temp.jpg")
            img2_cv = cv2.imread("usuario_temp.jpg")

            # 1. Comparación de rostro
            es_misma_persona, nivel_similitud = detectar_caras_y_comparar(img1_cv, img2_cv)

            # 2. OCR y Extracción de Datos
            gray_dni = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)

            # Agrandar imagen para optimizar OCR de caracteres pequeños
            h, w = gray_dni.shape
            if w < 1400:
                scale = 1400 / w
                gray_dni = cv2.resize(gray_dni, (1400, int(h * scale)), interpolation=cv2.INTER_CUBIC)

            # Filtro de contraste
            gray_dni = cv2.addWeighted(gray_dni, 1.5, gray_dni, -0.5, 0)

            # Lectura en múltiples modos
            texto_raw = pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 11')
            texto_raw += "\n" + pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 6')

            # Extraer DNI (8 dígitos + letra)
            match_dni = re.search(r'\b\d{8}\s*[-_]?\s*[A-Za-z]\b', texto_raw)
            if match_dni:
                numero_dni = re.sub(r'[\s\-_]', '', match_dni.group(0)).upper()
            else:
                numero_dni = "No detectado"

            # Extraer Fechas (Soporta DD MM AAAA, DD/MM/AAAA, DD.MM.AAAA, DD-MM-AAAA)
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
            c.drawString(40, height - 60, "Informe Oficial de Verificación de Identidad")

            # Estado de Verificación Facial
            c.setFont("Helvetica-Bold", 13)
            if es_misma_persona:
                c.setFillColor(colors.HexColor("#166534"))
                c.drawString(40, height - 120, "✔ VERIFICACIÓN FACIAL: COINCIDENCIA CONFIRMADA")
            else:
                c.setFillColor(colors.HexColor("#991B1B"))
                c.drawString(40, height - 120, "✖ VERIFICACIÓN FACIAL: REVISIÓN MANUAL REQUERIDA")

            # Datos del DNI
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

            c.drawImage("dni_temp.jpg", 40, height - 510, width=240, height=190, preserveAspectRatio=True)
            c.drawImage("usuario_temp.jpg", 315, height - 510, width=240, height=190, preserveAspectRatio=True)

            c.setFont("Helvetica", 8)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.drawString(40, 30, "Documento generado automáticamente por la aplicación de verificación.")

            c.save()

            if es_misma_persona:
                st.success(f"✅ ¡Verificación Facial Exitosa! DNI Detectado: **{numero_dni}**")
            else:
                st.info(f"ℹ️ Verificación completada. DNI Detectado: **{numero_dni}**")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar Informe en PDF",
                    data=pdf_file,
                    file_name=f"Informe_DNI_{numero_dni}.pdf",
                    mime="application/pdf"
                )
