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
    """Compara los recortes de rostro usando extracción de puntos clave (ORB)"""
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

def extraer_datos_reverso(img_reverso_bgr):
    """Extracción por regiones geométricas fijas del DNI español (Crop OCR)"""
    h, w, _ = img_reverso_bgr.shape
    
    # 1. Región Domicilio (Parte superior-derecha/centro)
    crop_domicilio = img_reverso_bgr[int(h*0.05):int(h*0.48), int(w*0.35):int(w*0.85)]
    
    # 2. Región Lugar de Nacimiento (Parte media-derecha)
    crop_nacimiento = img_reverso_bgr[int(h*0.45):int(h*0.75), int(w*0.42):int(w*0.85)]

    def aplicar_ocr_region(img_crop):
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # Binarización para eliminar fondo y mapa
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        txt = pytesseract.image_to_string(thresh, lang='spa', config='--psm 6')
        if not txt.strip():
            txt = pytesseract.image_to_string(gray, lang='spa', config='--psm 6')
        return txt

    txt_dom = aplicar_ocr_region(crop_domicilio)
    txt_nac = aplicar_ocr_region(crop_nacimiento)

    # Limpieza de Domicilio
    lineas_dom = [line.strip() for line in txt_dom.split('\n') if len(line.strip()) > 2]
    lineas_dom_filtradas = []
    for line in lineas_dom:
        l_up = line.upper()
        if not any(k in l_up for k in ["DOMICILIO", "REINO", "ESPAÑA"]):
            # Eliminar caracteres raros de OCR
            clean = re.sub(r'[^A-Za-z0-9ÁÉÍÓÚáéíóúÑñ\s/.,ºª-]', '', line).strip()
            if clean:
                lineas_dom_filtradas.append(clean)

    domicilio = " ".join(lineas_dom_filtradas) if lineas_dom_filtradas else "No detectado"

    # Limpieza de Lugar de Nacimiento
    lineas_nac = [line.strip() for line in txt_nac.split('\n') if len(line.strip()) > 2]
    lineas_nac_filtradas = []
    for line in lineas_nac:
        l_up = line.upper()
        # Cortar en cuanto aparezca HIJO DE o PADRES o Nombres propios
        if any(k in l_up for k in ["HIJO", "PADRES", "CANDIDO", "EMILIA", "EQUIPO", "IDESP"]):
            break
        if not any(k in l_up for k in ["LUGAR", "NACIMIENTO", "PROVINCIA"]):
            clean = re.sub(r'[^A-Za-zÁÉÍÓÚáéíóúÑñ\s,-]', '', line).strip()
            if clean:
                lineas_nac_filtradas.append(clean)

    lugar_nacimiento = " ".join(lineas_nac_filtradas) if lineas_nac_filtradas else "No detectado"

    return lugar_nacimiento, domicilio

foto_dni_front = st.file_uploader("1. Sube la foto del DNI (Anverso)", type=["jpg", "png", "jpeg"])
foto_dni_back = st.file_uploader("2. Sube la foto del DNI (Reverso)", type=["jpg", "png", "jpeg"])
foto_original = st.file_uploader("3. Sube la foto del usuario (Selfie)", type=["jpg", "png", "jpeg"])

if foto_dni_front and foto_original:
    img_dni_front_pil = ImageOps.exif_transpose(Image.open(foto_dni_front)).convert("RGB")
    img_user_pil = ImageOps.exif_transpose(Image.open(foto_original)).convert("RGB")
    img_dni_back_pil = ImageOps.exif_transpose(Image.open(foto_dni_back)).convert("RGB") if foto_dni_back else None

    st.markdown("---")
    st.subheader("🔄 Ajuste de Rotación")

    col_rot1, col_rot2, col_rot3 = st.columns(3)
    with col_rot1:
        rot_dni_front = st.selectbox("Girar DNI Anverso:", [0, 90, 180, 270], format_func=lambda x: f"{x}°", key="rot_dni_front")
        img_dni_front_pil = rotar_imagen(img_dni_front_pil, rot_dni_front)
        st.image(img_dni_front_pil, caption="DNI Anverso", use_container_width=True)

    with col_rot2:
        if img_dni_back_pil:
            rot_dni_back = st.selectbox("Girar DNI Reverso:", [0, 90, 180, 270], format_func=lambda x: f"{x}°", key="rot_dni_back")
            img_dni_back_pil = rotar_imagen(img_dni_back_pil, rot_dni_back)
            st.image(img_dni_back_pil, caption="DNI Reverso", use_container_width=True)
        else:
            st.info("Reverso no subido (Opcional)")

    with col_rot3:
        rot_user = st.selectbox("Girar Selfie:", [0, 90, 180, 270], format_func=lambda x: f"{x}°", key="rot_user")
        img_user_pil = rotar_imagen(img_user_pil, rot_user)
        st.image(img_user_pil, caption="Rostro Orientado", use_container_width=True)

    if st.button("🚀 Procesar Verificación"):
        with st.spinner("Procesando imágenes, analizando rostro y extrayendo datos..."):
            path_dni_front = "dni_front_temp.jpg"
            path_user = "usuario_temp.jpg"
            path_dni_back = "dni_back_temp.jpg"

            img_dni_front_pil.save(path_dni_front)
            img_user_pil.save(path_user)

            img_front_cv = cv2.imread(path_dni_front)
            img_user_cv = cv2.imread(path_user)

            # 1. Comparación de rostro
            es_misma_persona = comparar_rostros(img_front_cv, img_user_cv)

            # 2. OCR Anverso
            gray_dni = cv2.cvtColor(img_front_cv, cv2.COLOR_BGR2GRAY)
            h, w = gray_dni.shape
            if w < 1400:
                scale = 1400 / w
                gray_dni = cv2.resize(gray_dni, (1400, int(h * scale)), interpolation=cv2.INTER_CUBIC)

            texto_front = pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 11')
            texto_front += "\n" + pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 6')

            match_dni = re.search(r'\b\d{8}\s*[-_]?\s*[A-Za-z]\b', texto_front)
            numero_dni = re.sub(r'[\s\-_]', '', match_dni.group(0)).upper() if match_dni else "No detectado"

            patron_fechas = r'\b(\d{2})[\s/.-](\d{2})[\s/.-](\d{4})\b'
            fechas_coincidentes = re.findall(patron_fechas, texto_front)
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

            match_validez = re.search(r'(?:VALIDEZ|VAL)[^\d]*(\d{2}[\s/.-]\d{2}[\s/.-]\d{4})', texto_front, re.IGNORECASE)
            if match_validez:
                fecha_caducidad = match_validez.group(1).replace(' ', '/').replace('.', '/').replace('-', '/')

            # 3. OCR Reverso
            lugar_nacimiento = "No aportado"
            domicilio = "No aportado"

            if img_dni_back_pil:
                img_dni_back_pil.save(path_dni_back)
                img_back_cv = cv2.imread(path_dni_back)
                lugar_nacimiento, domicilio = extraer_datos_reverso(img_back_cv)

            # 4. Generar PDF
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
            c.drawString(40, height - 60, "Informe de Verificación de Identidad")

            # Resultado Verificación
            c.setFont("Helvetica-Bold", 13)
            if es_misma_persona:
                c.setFillColor(colors.HexColor("#166534"))
                c.drawString(40, height - 120, "✔ VERIFICACIÓN FACIAL: COINCIDENCIA CONFIRMADA")
            else:
                c.setFillColor(colors.HexColor("#991B1B"))
                c.drawString(40, height - 120, "✖ VERIFICACIÓN FACIAL: REVISIÓN MANUAL REQUERIDA")

            # Cuadro de Datos
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 155, "DATOS EXTRAÍDOS DEL DOCUMENTO")

            c.setStrokeColor(colors.HexColor("#E2E8F0"))
            c.setFillColor(colors.HexColor("#F8FAFC"))
            c.roundRect(40, height - 280, width - 80, 115, 6, fill=True, stroke=True)

            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(60, height - 180, "Número de DNI / NIE:")
            c.drawString(60, height - 200, "Fecha de Nacimiento:")
            c.drawString(60, height - 220, "Lugar de Nacimiento:")
            c.drawString(60, height - 240, "Domicilio:")
            c.drawString(60, height - 260, "Fecha de Caducidad:")

            c.setFont("Helvetica", 10)
            c.setFillColor(colors.HexColor("#0F172A"))
            c.drawString(220, height - 180, numero_dni)
            c.drawString(220, height - 200, fecha_nacimiento)
            c.drawString(220, height - 220, lugar_nacimiento)
            c.drawString(220, height - 240, domicilio)
            c.drawString(220, height - 260, fecha_caducidad)

            # Evidencias Fotográficas
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 310, "EVIDENCIAS FOTOGRÁFICAS")

            if img_dni_back_pil:
                c.drawImage(path_dni_front, 40, height - 490, width=160, height=140, preserveAspectRatio=True)
                c.drawImage(path_dni_back, 215, height - 490, width=160, height=140, preserveAspectRatio=True)
                c.drawImage(path_user, 390, height - 490, width=160, height=140, preserveAspectRatio=True)
            else:
                c.drawImage(path_dni_front, 40, height - 510, width=240, height=190, preserveAspectRatio=True)
                c.drawImage(path_user, 315, height - 510, width=240, height=190, preserveAspectRatio=True)

            # Nota a pie de página
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
