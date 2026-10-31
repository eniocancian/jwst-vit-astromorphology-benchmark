import streamlit as st
import requests

st.set_page_config(
    page_title="JWST Astromorphology Explorer",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 JWST Astromorphology & ViT-Tiny Explorer")
st.markdown(
    "Explore os embeddings latentes extraídos pelo **ViT-Tiny** de patches do James Webb Space Telescope (JWST NIRCam - Sagittarius C) "
    "e obtenha análises astrofísicas inteligentes geradas pela integração com o **Gemini**."
)

st.sidebar.header("🎛️ Configuração da Análise")
patch_id_input = st.sidebar.text_input("ID do Patch (ex: patch_001)", value="patch_001")

# Aponta para a nossa porta 8001 que acabou de subir
API_URL = "http://127.0.0.1:8001/analisar-patch"

if st.sidebar.button("🚀 Analisar Patch com IA"):
    with st.spinner("Consultando ViT-Tiny e Gemini API..."):
        try:
            response = requests.post(API_URL, json={"patch_id": patch_id_input})
            if response.status_code == 200:
                data = response.json()
                st.success("Análise concluída com sucesso!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Metadados do Vetor Latente")
                    st.metric(label="Identificador do Patch", value=data["patch_id"])
                    st.metric(label="Dimensão do Embedding (ViT-Tiny)", value=data["embedding_dim"])
                    
                with col2:
                    st.subheader("🌌 Análise Astrofísica (Gemini)")
                    st.info(data["analise_gemini"])
                    
            else:
                st.error(f"Erro na API: {response.json().get('detail', 'Erro desconhecido')}")
        except Exception as e:
            st.error(f"Não foi possível conectar à API FastAPI na porta 8001. Erro: {e}")

st.markdown("---")
st.markdown("💡 *Mantenha o terminal com o `app_completo.py` rodando e abra esta interface via Streamlit.*")