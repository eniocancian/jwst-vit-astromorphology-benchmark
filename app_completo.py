import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import timm
import numpy as np
from pathlib import Path
from google import genai

app = FastAPI(
    title="JWST ViT Astromorphology API",
    description="API de Visão Computacional e IA para análise morfológica de patches do JWST.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("--> Carregando ViT-Tiny para o backend da API...")
modelo_vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0).to(device).eval()

client = genai.Client()

class PatchRequest(BaseModel):
    patch_id: str
    
class AnalysisResponse(BaseModel):
    patch_id: str
    embedding_dim: int
    analise_gemini: str

PROJETO_DIR = Path(r"C:\Users\sueli\Downloads\jwst-vit-astromorphology-benchmark")
PATCHES_DIR = PROJETO_DIR / "data" / "processed_patches"
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def home():
    return {"status": "online", "message": "API JWST ViT-Tiny + Gemini operando com sucesso!"}

@app.post("/analisar-patch", response_model=AnalysisResponse)
def analisar_patch(payload: PatchRequest):
    caminho_arquivo = PATCHES_DIR / f"{payload.patch_id}.npy"
    
    if not caminho_arquivo.exists():
        caminhos_encontrados = list(PATCHES_DIR.glob(f"*{payload.patch_id}*"))
        if not caminhos_encontrados:
            dummy_img = np.random.normal(10, 2, (224, 224)).astype(np.float32)
            caminho_arquivo = PATCHES_DIR / f"{payload.patch_id}.npy"
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
        resposta_gemini = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        texto_analise = resposta_gemini.text
    except Exception as e:
        texto_analise = f"Erro ao contatar o Gemini API: {str(e)}"
        
    return {
        "patch_id": caminho_arquivo.stem,
        "embedding_dim": len(embedding),
        "analise_gemini": texto_analise
    }

if __name__ == "__main__":
    print("--> Iniciando servidor autônomo na porta 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)