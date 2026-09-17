"""Offline and replay-time policy distillation utilities."""

from __future__ import annotations

import torch


def masked_kl(student_q, teacher_q, legal_mask, *, temperature: float = 1.0):
    """KL(teacher || student) over only physically legal actions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    legal_mask = legal_mask.to(dtype=torch.bool)
    if legal_mask.ndim != 2 or legal_mask.shape != student_q.shape:
        raise ValueError("legal mask must match the Q-value matrix")
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("every distillation row needs one legal action")
    floor = torch.finfo(student_q.dtype).min
    student_log = torch.log_softmax(
        (student_q / temperature).masked_fill(~legal_mask, floor), dim=1)
    teacher_log = torch.log_softmax(
        (teacher_q / temperature).masked_fill(~legal_mask, floor), dim=1)
    teacher_probability = teacher_log.exp()
    terms = torch.where(
        legal_mask, teacher_probability * (teacher_log - student_log),
        torch.zeros_like(student_log),
    )
    return terms.sum(dim=1).mean() * temperature * temperature
