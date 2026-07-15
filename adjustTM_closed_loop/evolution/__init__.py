"""Phase-1 baseline evolution without generative image teachers."""

from .adjudication import AdjudicationConfig, adjudicate_evolution
from .candidate_selection import CandidateSelector, SelectionConfig, load_candidate_scores
from .policy import PolicyPrediction, RidgeAlphaPolicy, fit_policy_from_manifest
from .schemas import CandidateScore, DistributionStatus, TeacherRecord, TeacherSelection
from .score_merge import merge_candidate_scores
from .teacher_manifest import build_teacher_manifest, load_teacher_manifest

__all__ = [
    "AdjudicationConfig",
    "CandidateScore",
    "CandidateSelector",
    "DistributionStatus",
    "PolicyPrediction",
    "RidgeAlphaPolicy",
    "SelectionConfig",
    "TeacherRecord",
    "TeacherSelection",
    "adjudicate_evolution",
    "build_teacher_manifest",
    "fit_policy_from_manifest",
    "load_candidate_scores",
    "load_teacher_manifest",
    "merge_candidate_scores",
]
