"""Compatibility imports for the unified Capability boundary.

SForge V1 has no separate Skill/Tool manager. New code should import from
`harness.capability` directly.
"""

from harness.capability import (
    AdmissionPolicy,
    Capability,
    CapabilityRegistry,
    DefaultAdmissionPolicy,
    FunctionCapability,
)

__all__ = [
    "AdmissionPolicy",
    "Capability",
    "CapabilityRegistry",
    "DefaultAdmissionPolicy",
    "FunctionCapability",
]
