"""Normalizacao Astrofisica e Tiling de Mosaicos FITS."""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

def processar_e_fatiar_mosaico(fits_path: Path, output_dir: Path, patch_size=224, stride=168):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--> Processando: {fits_path.name}")
    with fits.open(fits_path) as hdul:
        dados_sci = hdul['SCI'].data
    dim_y, dim_x = dados_sci.shape
    dados_limpos = np.nan_to_num(dados_sci, nan=0.0, posinf=0.0, neginf=0.0)
    intervalo = ZScaleInterval()
    vmin, vmax = intervalo.get_limits(dados_limpos)
    dados_truncados = np.clip(dados_limpos, vmin, vmax)
    normalizador = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    resultado_norm = normalizador(dados_truncados)
    if np.ma.isMaskedArray(resultado_norm):
        img_norm = resultado_norm.filled(0.0).astype(np.float32)
    else:
        img_norm = np.asarray(resultado_norm, dtype=np.float32)
    patches_cont = 0
    for y in range(0, dim_y - patch_size, stride):
        for x in range(0, dim_x - patch_size, stride):
            patch = img_norm[y:y+patch_size, x:x+patch_size]
            if np.count_nonzero(patch) > (0.6 * patch_size * patch_size):
                np.save(output_dir / f"patch_{y}_{x}.npy", patch)
                patches_cont += 1
    print(f"Concluido: {patches_cont} patches salvos em {output_dir}")
    return patches_cont

if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent.parent
    caminhos = list((raiz / "data" / "raw_fits").rglob("*_i2d.fits"))
    if caminhos:
        processar_e_fatiar_mosaico(caminhos[0], raiz / "data" / "processed_patches")
