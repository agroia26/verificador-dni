import streamlit as st
from PIL import Image
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from deepface import DeepFace
import os

st.set_page_config(page_title="Verificación DNI", layout="centered")
st.title("🆔 Verificador de DNI y Comparativa Facial")

foto_dni = st.file_uploader("1. Sube la foto del DNI", type=["jpg", "png", "jpeg"])
foto_original = st.file_uploader("2. Sube la foto del usuario (Selfie)", type=["jpg", "png", "jpeg"])

if foto_dni and foto_original:
    col1, col2 = st.columns(2)
    with col1:
        st.image(foto_dni, caption="DNI", use_container_width=True)
    with col2:
        st.image(foto_original, caption="Rostro Usuario", use_container_width=True)

    if st.button("🚀 Procesar Verificación"):
        with st.spinner("Analizando rostros y extrayendo datos..."):
            with open("dni_temp.jpg", "wb") as f:
                f.write(foto_dni.getbuffer())
            with open("usuario_temp.jpg", "wb") as f:
                f.write(foto_original.getbuffer())

            # Comparación facial
            try:
                resultado = DeepFace.verify(img1_path="dni_temp.jpg", img2_path="usuario_temp.jpg", enforce_detection=False)
                es_misma_persona = resultado.get("verified", False)
            except Exception:
                es_misma_persona = False

            # Extracción de texto
            imagen_dni = Image.open("dni_temp.jpg")
            texto_dni = pytesseract.image_to_string(imagen_dni, lang='spa')

            # Generación de PDF
            pdf_path = "reporte_dni.pdf"
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 750, "Reporte de Identificacion y Verificación Facial")
            
            c.drawImage("dni_temp.jpg", 50, 520, width=220, preserveAspectRatio=True)
            c.drawImage("usuario_temp.jpg", 300, 520, width=220, preserveAspectRatio=True)
            
            c.setFont("Helvetica-Bold", 12)
            coincidencia = "COINCIDEN" if es_misma_persona else "NO COINCIDEN"
            c.drawString(50, 480, f"Resultado Comparacion Facial: {coincidencia}")
            
            c.drawString(50, 450, "Texto Detectado en el DNI:")
            c.setFont("Helvetica", 10)
            y = 430
            for linea in texto_dni.split('\n'):
                if linea.strip():
                    c.drawString(50, y, linea.strip()[:80])
                    y -= 15
                    if y < 50:
                        break
            c.save()

            if es_misma_persona:
                st.success("✅ ¡Los rostros coinciden exitosamente!")
            else:
                st.error("❌ Los rostros no parecen coincidir.")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Descargar Informe en PDF",
                    data=pdf_file,
                    file_name="Verificacion_DNI.pdf",
                    mime="application/pdf"
                )
