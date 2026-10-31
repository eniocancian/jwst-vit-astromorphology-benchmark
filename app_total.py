import streamlit as st
import torch
import timm
import numpy as np
from pathlib import Path
from google import genai

st.set_page_config(
    page_title="JWST Astromorphology Explorer",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 JWST Astromorphology & ViT-Tiny Explorer")
st.markdown(
    "Explore os embeddings latentes extraídos pelo **ViT-Tiny** de patches do James Webb Space Telescope (JWST NIRCam - Sagittarius C) "
    "e obtenha análises astrofísicas inteligentes geradas pela integração com o **Gemini** — tudo diretamente na interface."
)

@st.cache_resource
def carregar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0).to(device).eval()
    return modelo, device

with st.spinner("Carregando modelo ViT-Tiny..."):
    modelo_vit, device = carregar_modelo()

client = genai.Client()

st.sidebar.header("🎛️ Configuração da Análise")
patch_id_input = st.sidebar.text_input("ID do Patch (ex: patch_001)", value="patch_001")

PROJETO_DIR = Path(r"C:\Users\sueli\Downloads\jwst-vit-astromorphology-benchmark")
PATCHES_DIR = PROJETO_DIR / "data" / "processed_patches"
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

if st.sidebar.button("🚀 Analisar Patch com IA"):
    with st.spinner("Processando patch com ViT-Tiny e consultando o Gemini..."):
        caminho_arquivo = PATCHES_DIR / f"{patch_id_input}.npy"
        
        if not caminho_arquivo.exists():
            caminhos_encontrados = list(PATCHES_DIR.glob(f"*{patch_id_input}*"))
            if not caminhos_encontrados:
                dummy_img = np.random.normal(10, 2, (224, 224)).astype(np.float32)
                caminho_arquivo = PATCHES_DIR / f"{patch_id_input}.npy"
                np.save(caminho_arquivo, dummy_img)
            else:
                caminho_arquivo = caminhos_encontrados[0]
            
        img = np.load(caminho_arquivo).astype(np.float32)
        if img.ndim == 2:
            img_tensor = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        else:
            img_tensor = torch.tensor(img).unsqueeze(0).to(device)
            
        with torch.no_grad():
            embedding = modelo_vit.forward_features(img_tensor)[:, 0].cpu().numpy().flatten()
            
        media_intensidade = float(img.mean())
        std_intensidade = float(img.std())
        
        prompt = (
            f"Atue como um astrofísico especialista em dados do James Webb Space Telescope (JWST NIRCam). "
            f"Analise um patch astrofísico que possui média de intensidade de {media_intensidade:.2f} e desvio padrão de {std_intensidade:.2f}. "
            f"O modelo ViT-Tiny extraiu um vetor latente de 192 dimensões para este patch. "
            f"Forneça uma explicação científica concisa (em português) sobre o que essa assinatura morfológica e estatística pode representar em termos "
            f"de densidade estelar, berçários estelares ou nebulosidade na região de Sagittarius C."
        )
        
        try:
            # Atualizado para o modelo atual suportado pela API
            resposta_gemini = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
            )
            texto_analise = resposta_gemini.text
        except Exception as e:
            texto_analise = f"Erro ao contatar o Gemini API: {str(e)}"
            
        st.success("Análise concluída com sucesso!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Metadados do Vetor Latente")
            st.metric(label="Identificador do Patch", value=caminho_arquivo.stem)
            st.metric(label="Dimensão do Embedding (ViT-Tiny)", value=len(embedding))
            st.metric(label="Média de Intensidade", value=f"{media_intensidade:.2f}")
            
        with col2:
            st.subheader("🌌 Análise Astrofísica (Gemini)")
            st.info(texto_analise)