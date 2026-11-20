"""Unchanged frozen Roost composition encoding, without a pandas dataset."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def encode_formula(formula, space_group=None, device="cpu"):
    import torch
    import torch_scatter
    from pymatgen.core import Composition
    sys.path.insert(0, str(ROOT / "models/official/MatterVial-MP-gap"))
    from mattervial.packages.roost.roost.model import Roost
    composition = Composition(formula)
    amounts = composition.get_el_amt_dict()
    elements, total = list(amounts), sum(amounts.values())
    if total <= 0 or any(value <= 0 for value in amounts.values()):
        raise ValueError("formula must contain positive element amounts")
    if space_group is not None and (type(space_group) is not int or not 1 <= space_group <= 230):
        raise ValueError("space_group must be an integer in 1..230")
    base = ROOT / "models/official/MatterVial-MP-gap/mattervial/featurizers/custom_models"
    path = base / "mpgap_checkpoint-r0.pth.tar"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = json.loads((base / "matscholar-embedding.json").read_text())
    n = len(elements)
    device = torch.device(device)
    torch.set_num_threads(2)
    inputs = (torch.tensor([[amounts[e] / total] for e in elements], dtype=torch.float32),
              torch.tensor([embeddings[e] for e in elements], dtype=torch.float32),
              torch.tensor([i for i in range(n) for _ in range(n)], dtype=torch.long),
              torch.tensor(list(range(n)) * n, dtype=torch.long),
              torch.zeros(n, dtype=torch.long))
    inputs = tuple(value.to(device) for value in inputs)
    encoder = Roost(**checkpoint["model_params"], device=device).to(device).eval()
    encoder.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        hidden = encoder.material_nn(*inputs)[0].cpu().tolist()
        raw = float(list(encoder(*inputs))[0][0, 0])
    normalizer = checkpoint["normalizer_dict"]["band_gap"]
    scalar = raw * float(normalizer["std"]) + float(normalizer["mean"])
    fractions = [0.] * 118
    for element in composition.elements:
        fractions[element.Z - 1] = amounts[str(element)] / total
    return dict(formula=formula, composition=fractions + hidden + [scalar], space_group=space_group,
                encoder="unchanged pretrained Roost64 plus118 fractions plus1 scalar",
                encoder_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
