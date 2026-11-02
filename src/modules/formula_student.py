"""Formula student with report-supervised shared projections and density readout."""
import torch


class FormulaStudent(torch.nn.Module):
    def __init__(self, source, update_shared=True):
        super().__init__()
        self.source = source
        for parameter in source.parameters():
            parameter.requires_grad_(False)
        names = ('composition_projection', 'fusion', 'report_head') if update_shared else ('report_head',)
        for name in names:
            for parameter in getattr(source, name).parameters():
                parameter.requires_grad_(True)

    @staticmethod
    def formula_input(inputs):
        return torch.cat((inputs[..., :183], inputs.new_zeros(*inputs.shape[:-1], 399)), dim=-1)

    def report(self, inputs, bounds=(.05, 2.5)):
        return self.source.report(self.formula_input(inputs), bounds)

    def aggregate(self, inputs):
        return self.source.aggregate(self.formula_input(inputs))


def component_kl(teacher_mu, teacher_sigma, student_mu, student_sigma):
    """KL of lognormal components; row-weighted gradient matches mixture cross entropy."""
    tm, ts, sm, ss = [value.double() for value in
                       (teacher_mu, teacher_sigma, student_mu, student_sigma)]
    return torch.log(ss / ts) + (ts.square() + (tm - sm).square()) / (2 * ss.square()) - .5
