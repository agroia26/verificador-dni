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

def limpiar_texto_ruido(texto_raw, max_lineas=3):
    """Filtra y elimina de forma estricta líneas con basura/ruido de OCR"""
    lineas_limpias = []
    for line in texto_raw.split('\n'):
        clean = line.strip()
        if not clean:
            continue
        
        # Ignorar encabezados conocidos
        upper = clean.upper()
        if any(k in upper for k in ["DOMICILIO", "LUGAR DE NACIMIENTO", "PADRES", "HIJO", "EQUIPO", "IDESP", "REINO"]):
            continue

        # Mantener solo caracteres alfanuméricos y puntuación básica
        clean_text = re.sub(r'[^A-Za-z0-9ÁÉÍÓÚáéíóúÑñ\s/.,ºª-]', '', clean).strip()
        palabras = clean_text.split()

        # Filtrar líneas basura (por ejemplo, líneas llenas de letras sueltas 'a b c d')
        palabras_validas = [p for p in palabras if len(p) > 1 or p.isdigit()]
        if len(palabras_validas) >= 1 and len(clean_text) >= 3:
            lineas_limpias.append(clean_text)

        if len(lineas_limpias) >= max_lineas:
            break

    return "\n".join(lineas_limpias)

def extraer_datos_reverso(img_reverso_bgr):
    """Procesamiento OCR mejorado con limpieza adaptativa"""
    h, w, _ = img_reverso_bgr.shape

    # Recortes más holgados para no perder texto por bordes de imagen
    crop_dom = img_reverso_bgr[int(h*0.02):int(h*0.38), int(w*0.30):int(w*0.90)]
    crop_nac = img_reverso_bgr[int(h*0.38):int(h*0.65), int(w*0.30):int(w*0.90)]

    def procesar_crop(crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # Binarización suave
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        txt = pytesseract.image_to_string(thresh, lang='spa', config='--psm 6')
        if len(txt.strip()) < 5:
            txt = pytesseract.image_to_string(gray, lang='spa', config='--psm 6')
        return txt

    raw_dom = procesar_crop(crop_dom)
    raw_nac = procesar_crop(crop_nac)

    domicilio_str = limpiar_texto_ruido(raw_dom, max_lineas=3)
    lugar_nac_str = limpiar_texto_ruido(raw_nac, max_lineas=2)

    return lugar_nac_str, domicilio_str

# ------------------- INTERFAZ STREAMLIT -------------------

foto_dni_front = st.file_uploader("1. Sube la foto del DNI (Anverso)", type=["jpg", "png", "jpeg"])
foto_dni_back = st.file_uploader("2. Sube la foto del DNI (Reverso)", type=["jpg", "png", "jpeg"])
foto_original = st.file_uploader("3. Sube la foto del usuario (Selfie)", type=["jpg", "png", "jpeg"])

if foto_dni_front and foto_original:
    img_dni_front_pil = ImageOps.exif_transpose(Image.open(foto_dni_front)).convert("RGB")
    img_user_pil = ImageOps.exif_transpose(Image.open(foto_original)).convert("RGB")
    img_dni_back_pil = ImageOps.exif_transpose(Image.open(foto_dni_back)).convert("RGB") if foto_dni_back else None

    st.markdown("---")
    st.subheader("🔄 Ajuste de Orientación")

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

    st.markdown("---")

    if st.button("🔍 Extraer Datos y Verificar Rostro"):
        with st.spinner("Analizando imágenes..."):
            path_dni_front = "dni_front_temp.jpg"
            path_user = "usuario_temp.jpg"
            path_dni_back = "dni_back_temp.jpg"

            img_dni_front_pil.save(path_dni_front)
            img_user_pil.save(path_user)

            img_front_cv = cv2.imread(path_dni_front)
            img_user_cv = cv2.imread(path_user)

            # Comparar rostros
            st.session_state["es_misma_persona"] = comparar_rostros(img_front_cv, img_user_cv)

            # OCR Anverso
            gray_dni = cv2.cvtColor(img_front_cv, cv2.COLOR_BGR2GRAY)
            h, w = gray_dni.shape
            if w < 1400:
                scale = 1400 / w
                gray_dni = cv2.resize(gray_dni, (1400, int(h * scale)), interpolation=cv2.INTER_CUBIC)

            texto_front = pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 11')
            texto_front += "\n" + pytesseract.image_to_string(gray_dni, lang='spa', config='--psm 6')

            match_dni = re.search(r'\b\d{8}\s*[-_]?\s*[A-Za-z]\b', texto_front)
            st.session_state["numero_dni"] = re.sub(r'[\s\-_]', '', match_dni.group(0)).upper() if match_dni else ""

            patron_fechas = r'\b(\d{2})[\s/.-](\d{2})[\s/.-](\d{4})\b'
            fechas_coincidentes = re.findall(patron_fechas, texto_front)
            fechas_formateadas = [f"{d}/{m}/{a}" for d, m, a in fechas_coincidentes]

            st.session_state["fecha_nacimiento"] = fechas_formateadas[0] if len(fechas_formateadas) >= 1 else ""
            st.session_state["fecha_caducidad"] = fechas_formateadas[-1] if len(fechas_formateadas) >= 2 else ""

            match_validez = re.search(r'(?:VALIDEZ|VAL)[^\d]*(\d{2}[\s/.-]\d{2}[\s/.-]\d{4})', texto_front, re.IGNORECASE)
            if match_validez:
                st.session_state["fecha_caducidad"] = match_validez.group(1).replace(' ', '/').replace('.', '/').replace('-', '/')

            # OCR Reverso
            if img_dni_back_pil:
                img_dni_back_pil.save(path_dni_back)
                img_back_cv = cv2.imread(path_dni_back)
                nac, dom = extraer_datos_reverso(img_back_cv)
                st.session_state["lugar_nacimiento"] = nac
                st.session_state["domicilio"] = dom
            else:
                st.session_state["lugar_nacimiento"] = ""
                st.session_state["domicilio"] = ""

            st.session_state["procesado"] = True

    # Formulario de revisión de datos extraídos
    if st.session_state.get("procesado", False):
        st.subheader("📝 Revisión y Validación de Datos Extraídos")
        st.caption("Verifica o corrige cualquier dato antes de generar el informe en PDF:")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            dni_val = st.text_input("Número de DNI / NIE:", value=st.session_state.get("numero_dni", ""))
            fnac_val = st.text_input("Fecha de Nacimiento:", value=st.session_state.get("fecha_nacimiento", ""))
            fcad_val = st.text_input("Fecha de Caducidad:", value=st.session_state.get("fecha_caducidad", ""))

        with col_f2:
            lnac_val = st.text_area("Lugar de Nacimiento:", value=st.session_state.get("lugar_nacimiento", ""), height=80)
            dom_val = st.text_area("Domicilio:", value=st.session_state.get("domicilio", ""), height=100)

        es_misma_persona = st.session_state.get("es_misma_persona", False)

        if es_misma_persona:
            st.success("✔ Verificación Facial: Coincidencia Confirmada")
        else:
            st.warning("⚠️ Verificación Facial: Revisión Manual Requerida")

        if st.button("📄 Generar Informe PDF"):
            path_dni_front = "dni_front_temp.jpg"
            path_user = "usuario_temp.jpg"
            path_dni_back = "dni_back_temp.jpg"

            pdf_path = "informe_verificacion.pdf"
            c = canvas.Canvas(pdf_path, pagesize=A4)
            width, height = A4

            # Encabezado PDF
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

            # Cuadro de Datos Extraídos
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 155, "DATOS EXTRAÍDOS DEL DOCUMENTO")

            nac_lines = [l for l in lnac_val.split('\n') if l.strip()]
            dom_lines = [l for l in dom_val.split('\n') if l.strip()]
            if not nac_lines:
                nac_lines = ["No especificado"]
            if not dom_lines:
                dom_lines = ["No especificado"]

            box_height = 115 + (max(0, len(nac_lines) - 1) * 12) + (max(0, len(dom_lines) - 1) * 12)

            c.setStrokeColor(colors.HexColor("#E2E8F0"))
            c.setFillColor(colors.HexColor("#F8FAFC"))
            c.roundRect(40, height - 165 - box_height, width - 80, box_height, 6, fill=True, stroke=True)

            y_pos = height - 180

            # DNI
            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(60, y_pos, "Número de DNI / NIE:")
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica", 10)
            c.drawString(220, y_pos, dni_val)

            # Fecha Nacimiento
            y_pos -= 20
            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(60, y_pos, "Fecha de Nacimiento:")
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica", 10)
            c.drawString(220, y_pos, fnac_val)

            # Lugar de Nacimiento (Multilínea)
            y_pos -= 20
            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(60, y_pos, "Lugar de Nacimiento:")
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica", 10)
            for idx, line in enumerate(nac_lines):
                c.drawString(220, y_pos - (idx * 12), line)
            y_pos -= (len(nac_lines) - 1) * 12

            # Domicilio (Multilínea)
            y_pos -= 20
            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(60, y_pos, "Domicilio:")
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica", 10)
            for idx, line in enumerate(dom_lines):
                c.drawString(220, y_pos - (idx * 12), line)
            y_pos -= (len(dom_lines) - 1) * 12

            # Fecha Caducidad
            y_pos -= 20
            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(60, y_pos, "Fecha de Caducidad:")
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica", 10)
            c.drawString(220, y_pos, fcad_val)

            # Evidencias
            y_foto_section = height - 195 - box_height
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y_foto_section, "EVIDENCIAS FOTOGRÁFICAS")

            if os.path.exists(path_dni_back):
                c.drawImage(path_dni_front, 40, y_foto_section - 160, width=160, height=140, preserveAspectRatio=True)
                c.drawImage(path_dni_back, 215, y_foto_section - 160, width=160, height=140, preserveAspectRatio=True)
                c.drawImage(path_user, 390, y_foto_section - 160, width=160, height=140, preserveAspectRatio=True)
            else:
                c.drawImage(path_dni_front, 40, y_foto_section - 180, width=240, height=180, preserveAspectRatio=True)
                c.drawImage(path_user, 315, y_foto_section - 180, width=240, height=180, preserveAspectRatio=True)

            c.setFont("Helvetica-Oblique", 7.5)
            c.setFillColor(colors.HexColor("#64748B"))
            c.drawString(40, 30, "Aviso: Este documento no es oficial y se basa únicamente en un reconocimiento facial de características biométricas.")

            c.save()

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="⬇️ Descargar Informe en PDF",
                    data=pdf_file,
                    file_name=f"Informe_DNI_{dni_val}.pdf",
                    mime="application/pdf"
                )
