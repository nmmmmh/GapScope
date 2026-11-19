"""Pinned model/fit APIs and reconstruction of the new, self-contained checkpoints."""
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.dont_write_bytecode = True
import torch

HERE = Path(__file__).resolve().parent
MODULES = HERE / "modules"
VIEWS = ("formula_only", "formula_sg", "formula_structure", "full")
FEATURES = ("composition", "sg_features", "structure", "sg_mask", "structure_mask")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, values):
    if not values:
        return
    fields = list(dict.fromkeys(key for row in values for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def load(name, path):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    from types import SimpleNamespace
    paths = {
        "dual_task_model": "dual_task_model.py",
        "t10": "multimodal_runner.py",
        "fusion_model": "fusion_layers.py",
        "model_raw": "raw_bypass.py",
        "mixture_model": "density_heads.py",
        "t35_model": "symmetry_features.py",
        "model37": "formula_student.py",
        "model39": "structure_adaptation.py",
        "interaction_model": "integrated_fusion.py",
        "model": "preserved_fusion.py",
        "t43_model": "conditional_fusion.py",
    }
    m = {}
    for name, relative in paths.items():
        m[name] = load(name, MODULES / relative)
    m["Source"] = m["model_raw"].make_model_class(m["fusion_model"].ModalFusion)
    m["FormulaStudent"] = m["model37"].FormulaStudent
    m["interaction"] = m["interaction_model"].IntegratedFusion
    m["t35"] = m["t35_model"]
    def symmetry_table():
        rows = read_json(MODULES / "symmetry_table.json")["groups"]
        return torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    m["loader43"] = SimpleNamespace(symmetry_table=symmetry_table)
    return m


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError("T46 requires cuda:0")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda:0")


def source_from_state(m, state, device):
    model = m["Source"]("simple_concat", state["report_head.raw_beta"],
                        float(state["report_head.raw_feature_mean"]), float(state["report_head.raw_feature_std"]))
    model.load_state_dict(state)
    return model.to(device).eval()


def restore(m, cp, device):
    """Rebuild only T46 states, without searching any historical checkpoint."""
    if cp["schema"] != "t46_checkpoint_v1":
        raise ValueError("Expected a T46 checkpoint")
    state, node = cp["state_dict"], cp["node"]
    if node in ("t37_readout", "t37_modal_kd"):
        source_state = {key[len("source."):]: value for key, value in state.items() if key.startswith("source.")}
        model = m["FormulaStudent"](source_from_state(m, source_state, device), node != "t37_readout")
    elif node == "t39_structure":
        prefix = "formula_base.report_head."
        formula = m["Source"]("simple_concat", state[prefix + "raw_beta"],
                              float(state[prefix + "raw_feature_mean"]), float(state[prefix + "raw_feature_std"]))
        model = m["model39"].Adaptation(formula, m["t35"].SymmetryFusion("hybrid"),
                                       cp["basis"], "structure_projection").to(device)
        model.old_basis = cp["old_basis"].to(device)
    elif node in ("t42_M", "t43_formula"):
        base_state = state if node == "t42_M" else {
            key[len("base."):]: value for key, value in state.items() if key.startswith("base.")}
        branches = []
        for i in range(2):
            prefix = f"branches.{i}.source."
            source_state = {key[len(prefix):]: value for key, value in base_state.items() if key.startswith(prefix)}
            branches.append(m["FormulaStudent"](source_from_state(m, source_state, device), True))
        interaction = m["interaction"]("cross_attention", cp["beta"], cp["raw_feature_mean"], cp["raw_feature_std"], "M")
        model = m["model"].PreservedFusion(branches, interaction, "M").to(device)
        model.load_state_dict(base_state)
        if node == "t43_formula":
            interaction = m["interaction"]("cross_attention", cp["beta"], cp["raw_feature_mean"], cp["raw_feature_std"], "M")
            model = m["t43_model"].CenteredConditionalFusion(model, interaction, centered=False).to(device)
    else:
        raise ValueError(f"No inference endpoint for {node}")
    model.load_state_dict(state)
    return model.to(device).eval()


def components(model, inputs):
    with torch.no_grad():
        if hasattr(model, "components"):
            return model.components(inputs, (.05, 2.5))
        mu, sigma = model.report(inputs, (.05, 2.5))
        return torch.ones_like(mu).unsqueeze(-1), mu.double().unsqueeze(-1), sigma.double().unsqueeze(-1)

