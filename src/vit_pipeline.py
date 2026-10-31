"""Extracao de Embeddings e Explicabilidade com Vision Transformer."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from scipy.ndimage import zoom

class JWSTPatchesDataset(Dataset):
    def __init__(self, patches_dir: Path):
        self.arquivos = sorted(list(patches_dir.glob("*.npy")))
        if not self.arquivos:
            raise FileNotFoundError(f"Sem arquivos em: {patches_dir}")

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, idx):
        caminho = self.arquivos[idx]
        matriz = np.load(caminho).astype(np.float32)
        tensor_3c = torch.from_numpy(matriz).unsqueeze(0).repeat(3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor_3c - mean) / std, caminho.stem

def extrair_attention_rollout(modelo, tensor_imagem, device):
    modelo.eval()
    matrizes = []
    def hook_fn(module, inp, out):
        x = inp[0]
        B, N, C = x.shape
        num_heads = module.num_heads
        head_dim = C // num_heads
        scale = module.scale or (head_dim ** -0.5)
        qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1)
        matrizes.append(attn.detach().cpu())
    hooks = [b.attn.register_forward_hook(hook_fn) for b in modelo.blocks]
    with torch.no_grad():
        _ = modelo(tensor_imagem.unsqueeze(0).to(device))
    for h in hooks:
        h.remove()
    atencoes = [torch.mean(c, dim=1).squeeze(0) for c in matrizes]
    N = atencoes[0].size(-1)
    rollout = torch.eye(N)
    for c in atencoes:
        c_res = 0.5 * c + 0.5 * torch.eye(N)
        c_res = c_res / c_res.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(c_res, rollout)
    cls_attn = rollout[0, 1:].reshape(14, 14).numpy()
    exp = zoom(cls_attn, 224 / 14, order=1)
    return (exp - exp.min()) / (exp.max() - exp.min() + 1e-8)
