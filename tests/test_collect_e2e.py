"""Execução ponta a ponta do script de coleta, num mini-dataset sintético com o
MESMO formato do MVTec Screws (bbox = row, col, w, h, phi). Garante que o
main() roda de verdade e que as imagens de teste ficam fora da coleta."""
import csv
import json

import cv2
import numpy as np
import pytest

from experiments import collect_descriptors as cd

NUT_CLASS, SCREW_CLASS = 9, 1


def _make_dataset(root, n_images=6, n_test=2):
    img_dir = root / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(n_images):
        bg = np.empty((300, 400, 3), np.float32)
        bg[:] = (70, 150, 200)                      # madeira amarelada (BGR)
        bg += rng.normal(0, 6, bg.shape)
        img = np.clip(bg, 0, 255).astype(np.uint8)
        cv2.circle(img, (100, 150), 35, (140, 140, 140), -1)             # porca
        cv2.rectangle(img, (215, 142), (345, 158), (60, 60, 60), -1)     # parafuso
        name = f"img_{i:02d}.png"
        cv2.imwrite(str(img_dir / name), img)
        images.append({"id": i, "file_name": name, "width": 400, "height": 300})
        anns.append({"id": 2 * i, "image_id": i, "category_id": NUT_CLASS,
                     "bbox": [150, 100, 76, 76, 0.0]})
        anns.append({"id": 2 * i + 1, "image_id": i, "category_id": SCREW_CLASS,
                     "bbox": [150, 280, 140, 26, 0.0]})
    cats = [{"id": c, "name": f"c{c}"} for c in (SCREW_CLASS, NUT_CLASS)]
    full = {"categories": cats, "images": images, "annotations": anns}
    test_ids = set(range(n_images - n_test, n_images))
    test = {
        "categories": cats,
        "images": [im for im in images if im["id"] in test_ids],
        "annotations": [a for a in anns if a["image_id"] in test_ids],
    }
    (root / "all.json").write_text(json.dumps(full))
    (root / "test.json").write_text(json.dumps(test))
    return img_dir


def _run(monkeypatch, root, img_dir, *extra):
    argv = ["collect", "--annotations", str(root / "all.json"),
            "--test-annotations", str(root / "test.json"),
            "--images-dir", str(img_dir), "--out-dir", str(root / "out"), *extra]
    monkeypatch.setattr("sys.argv", argv)
    return cd.main()


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_roda_ponta_a_ponta_e_exclui_imagens_de_teste(tmp_path, monkeypatch, capsys):
    img_dir = _make_dataset(tmp_path)
    assert _run(monkeypatch, tmp_path, img_dir) == 0

    rows = _read(tmp_path / "out" / "component_descriptors.csv")
    used = {r["image"] for r in rows}
    assert used == {f"img_{i:02d}.png" for i in range(4)}      # 4 e 5 são do teste

    classes = {int(r["class_id"]) for r in rows}
    assert {NUT_CLASS, SCREW_CLASS} <= classes
    for r in rows:
        if int(r["class_id"]) == NUT_CLASS:
            assert r["group"] == "porca" and float(r["aspect_ratio"]) < 1.2
        if int(r["class_id"]) == SCREW_CLASS:
            assert r["group"] == "parafuso" and float(r["aspect_ratio"]) > 4

    assert (tmp_path / "out" / "descriptor_summary.csv").is_file()
    sizes = {int(r["class_id"]): r for r in _read(tmp_path / "out" / "gt_box_sizes.csv")}
    assert float(sizes[SCREW_CLASS]["lado_maior_mediana"]) == pytest.approx(140)
    assert "Imagens usadas: 4" in capsys.readouterr().out


def test_include_test_usa_todas_as_imagens(tmp_path, monkeypatch):
    img_dir = _make_dataset(tmp_path)
    assert _run(monkeypatch, tmp_path, img_dir, "--include-test") == 0
    rows = _read(tmp_path / "out" / "component_descriptors.csv")
    assert len({r["image"] for r in rows}) == 6


def test_limit_pega_amostra_espacada(tmp_path, monkeypatch):
    img_dir = _make_dataset(tmp_path)
    assert _run(monkeypatch, tmp_path, img_dir, "--limit", "2") == 0
    used = {r["image"] for r in _read(tmp_path / "out" / "component_descriptors.csv")}
    assert used == {"img_00.png", "img_03.png"}                 # extremos do conjunto elegível


def test_sem_json_de_teste_falha_alto_em_vez_de_vazar(tmp_path, monkeypatch, capsys):
    img_dir = _make_dataset(tmp_path)
    (tmp_path / "test.json").unlink()
    assert _run(monkeypatch, tmp_path, img_dir) == 1
    assert "--include-test" in capsys.readouterr().err


# --------------------------------------------------------------------------
# compare_channels: o CSV precisa registrar quantas imagens foram usadas
# (sem isso o resultado não é rastreável nem comparável entre execuções).
# --------------------------------------------------------------------------
def test_compare_channels_registra_n_imagens_no_csv(tmp_path, monkeypatch):
    from experiments import compare_channels as cc

    img_dir = _make_dataset(tmp_path)
    out = tmp_path / "canais.csv"
    monkeypatch.setattr("sys.argv", [
        "cc", "--annotations", str(tmp_path / "all.json"),
        "--images-dir", str(img_dir), "--limit", "3", "--out", str(out)])
    assert cc.main() == 0
    rows = _read(out)
    assert rows and all(int(r["n_imagens"]) == 3 for r in rows)
