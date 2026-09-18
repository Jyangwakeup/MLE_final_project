"""Versioned admission limits; v1 evidence keeps its original interpretation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    p95: float
    maximum: float


LEGACY = Limits(.25, .48)
CANDIDATE = Limits(.1, .3)
VERSION = 'task4-campaign-v2'
OBSERVED_VERSION = 'task4-campaign-v3'
ALIGNED_VERSION = 'task4-campaign-v4'
MODERN = (VERSION, OBSERVED_VERSION, ALIGNED_VERSION)


def limits(version, role='candidate'):
    if role not in ('candidate', 'reference', 'historical_reference'):
        raise ValueError('Unknown evaluation role')
    if version not in ('task4-campaign-v1', *MODERN):
        raise ValueError('Unknown campaign protocol')
    return CANDIDATE if version in MODERN and (role == 'candidate' or version == ALIGNED_VERSION and role == 'reference') else LEGACY


def timing_failures(summary, policy):
    return [key for key, bound in (
        ('act_p95_seconds', policy.p95), ('act_max_seconds', policy.maximum)
    ) if summary[key] > bound]
