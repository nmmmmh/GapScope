"""Minimal in-process support for single-element report formulas in Roost."""

import numpy as np
import torch


_original_getitem = None
_original_message_forward = None
_patched_getitem = None
_patched_message_forward = None


def enable_adapter():
    """Patch only the current process's CompositionData and MessageLayer."""
    global _original_getitem, _original_message_forward
    global _patched_getitem, _patched_message_forward

    from roost.roost.data import CompositionData, parse_roost
    from roost.roost.model import MessageLayer

    if _patched_getitem is not None:
        if (CompositionData.__getitem__ is not _patched_getitem or
                MessageLayer.forward is not _patched_message_forward):
            raise RuntimeError("Roost adapter classes changed after installation")
        return

    _original_getitem = CompositionData.__getitem__
    _original_message_forward = MessageLayer.forward

    def getitem_with_singletons(self, idx):
        cry_id, composition, target = self.df.iloc[idx]
        elements, weights = parse_roost(composition)
        if len(elements) != 1:
            return _original_getitem(self, idx)

        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_id} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_id} [{composition}] composition cannot be parsed into elements"
            )

        atom_weights = torch.Tensor([[1.0]])
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor([])
        nbr_fea_idx = torch.LongTensor([])
        if self.task == "regression":
            targets = torch.Tensor([float(target)])
        elif self.task == "classification":
            targets = torch.LongTensor([target])

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            targets,
            composition,
            cry_id,
        )

    def message_forward_with_singletons(
        self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx
    ):
        if self_fea_idx.numel() == 0:
            return elem_in_fea

        n = int(self_fea_idx.max().item()) + 1
        if n == elem_in_fea.shape[0]:
            return _original_message_forward(
                self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx
            )

        prefix = _original_message_forward(
            self,
            elem_weights[:n],
            elem_in_fea[:n],
            self_fea_idx,
            nbr_fea_idx,
        )
        return torch.cat((prefix, elem_in_fea[n:]), dim=0)

    _patched_getitem = getitem_with_singletons
    _patched_message_forward = message_forward_with_singletons
    CompositionData.__getitem__ = _patched_getitem
    MessageLayer.forward = _patched_message_forward
