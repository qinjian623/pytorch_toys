import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillLoss(nn.Module):
    def __init__(self, alpha, temperature, k=None):
        super(DistillLoss, self).__init__()
        self.alpha = alpha
        self.start_alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.TT = self.temperature * self.temperature
        self.ce = nn.CrossEntropyLoss()
        self.K = k

    def forward(self,
                student_out: torch.Tensor,
                teacher_out: torch.Tensor,
                label: torch.Tensor):
        if self.K is not None:
            _, index = teacher_out.topk(student_out.shape[1] - self.K, dim=1, largest=False)
            teacher_out[index] = 0.0  # TODO maybe uniform random value
        l0 = self.kl_loss(F.log_softmax(student_out / self.temperature, dim=1),
                          F.softmax(teacher_out / self.temperature, dim=1))
        l1 = self.ce(student_out, label)
        return l0 * self.alpha * self.TT + l1 * (1.0 - self.alpha)
