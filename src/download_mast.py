"""Download de Mosaicos FITS do JWST (NIRCam Level 3) via astroquery."""
from pathlib import Path
import warnings
from astropy.utils.exceptions import AstropyWarning
from astroquery.mast import Observations

warnings.simplefilter('ignore', category=AstropyWarning)

def baixar_mosaico_sgr_c(destino: Path, proposal_id="2221", limite=1):
    destino.mkdir(parents=True, exist_ok=True)
    print(f"--> Consultando MAST pelo Programa JWST {proposal_id} (Sagittarius C)...")
    tabela_obs = Observations.query_criteria(proposal_id=proposal_id, obs_collection="JWST")
    produtos = Observations.get_product_list(tabela_obs)
    mosaicos_i2d = Observations.filter_products(produtos, productType="SCIENCE", extension="fits")
    mascara = [str(f).endswith('_i2d.fits') for f in mosaicos_i2d['productFilename']]
    mosaicos_finais = mosaicos_i2d[mascara]
    print(f"    Mosaicos encontrados: {len(mosaicos_finais)}")
    print(f"--> Baixando {limite} mosaico(s) para {destino}...")
    Observations.download_products(mosaicos_finais[:limite], download_dir=str(destino), cache=True)
    print("Download concluido com sucesso!")

if __name__ == "__main__":
    destino_dados = Path(__file__).resolve().parent.parent / "data" / "raw_fits"
    baixar_mosaico_sgr_c(destino_dados)
