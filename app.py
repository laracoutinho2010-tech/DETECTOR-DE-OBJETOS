import gc
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Detector YOLO - Marketing", page_icon="👁️", layout="centered"
)


@st.cache_resource
def load_model():
  # Carrega o modelo YOLOv8 nano otimizado para CPU
  return YOLO("yolov8n.pt")


def main():
  st.title("🔍 Sistema de Identificação de Objetos (YOLO)")
  st.write("Faça upload de uma imagem para detecção automática de objetos.")

  model = load_model()

  uploaded_file = st.file_uploader(
      "Escolha uma imagem...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None:
    # Leitura da imagem nativa com PIL (já em formato RGB)
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem Original", use_container_width=True)

    if st.button("Executar Detecção"):
      with st.spinner("Processando..."):
        # O YOLO aceita objetos PIL diretamente e retorna em RGB por padrão
        results = model(image, conf=0.4)

        # O método .plot() retorna um array numpy em formato BGR por padrão,
        # mas convertemos direto para PIL com inversão ou tratamento seguro:
        res_bgr = results[0].plot()
        # Convertendo array BGR do plot do Ultralytics para RGB via PIL
        res_rgb = res_bgr[..., ::-1]
        result_image = Image.fromarray(res_rgb)

        st.image(
            result_image, caption="Resultado da Detecção", use_container_width=True
        )

      # Limpeza de memória do escopo atual
      del results
      del res_bgr
      del res_rgb
      gc.collect()


if __name__ == "__main__":
  main()
  
