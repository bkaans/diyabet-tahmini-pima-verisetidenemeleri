"""Veri mudahalesi deney sistemi testleri."""

from __future__ import annotations

from pathlib import Path

from makine_ogrenmesi.kaynak.maksimum_skor_arama import veri_butunlugu_ozeti
from makine_ogrenmesi.kaynak.veri_mudahale_deneyleri import veri_mudahale_deneyleri_calistir


def test_veri_mudahale_deneyi_ham_csvyi_koruyup_rapor_uretmeli(tmp_path: Path) -> None:
    proje_koku = Path(__file__).resolve().parents[1]
    veri_yolu = proje_koku / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"
    baslangic = veri_butunlugu_ozeti(veri_yolu)

    rapor = veri_mudahale_deneyleri_calistir(
        veri_yolu=veri_yolu,
        proje_koku=tmp_path,
        mod="quick",
        hedef_sinif_sayilari=(400,),
        model_adlari=("extra_trees_full",),
        max_varyant=1,
        n_jobs=1,
        artifact_yaz=False,
        word_raporu_yaz=False,
    )

    bitis = veri_butunlugu_ozeti(veri_yolu)
    assert bitis["sha256"] == baslangic["sha256"]
    assert rapor["en_iyi_sonuc"]["durum"] == "tamamlandi"
    assert "specificity" in rapor["en_iyi_sonuc"]["agresif_metrikler"]
    assert (
        tmp_path
        / "makine_ogrenmesi"
        / "raporlar"
        / "degerlendirme"
        / "veri_mudahale_leaderboard.json"
    ).exists()
