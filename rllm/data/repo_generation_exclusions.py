"""Known-inaccessible repository-generation benchmark instances.

The source datasets remain materialized verbatim for provenance. Runtime
training/evaluation filters these instances before scheduling any sandbox, so
an unavailable image cannot become an infrastructure retry storm.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DENOVOSWE_EXCLUDED_TASK_IDS = frozenset(
    {
        "Azure_msrest-for-python_pr247",
        "Azure_msrestazure-for-python_pr121",
        "Bouke_docx-mailmerge_pr43",
        "BrianPugh_autoregistry_pr35",
        "CFMTech_pytest-monitor_pr61",
        "Chilipp_docrep_pr19",
        "Deepwalker_trafaret_pr110",
        "DinoTools_python-overpy_pr63",
        "DreamLab_memoize_pr37",
        "Edinburgh-Genome-Foundry_dnachisel_pr59",
        "GehirnInc_python-jwt_pr33",
        "IDSIA_sacred_pr941",
        "JohannesBuchner_imagehash_pr166",
        "Kozea_cssselect2_pr24",
        "Kyligence_kylinpy_pr39",
        "LeapBeyond_scrubadub_pr54",
        "MacHu-GWU_sqlalchemy_mate-project_pr6",
        "MatthieuDartiailh_pyclibrary_pr81",
        "Mergifyio_daiquiri_pr91",
        "NOAA-ORR-ERD_lat_lon_parser_pr10",
        "NextThought_sphinxcontrib-programoutput_pr41",
        "OpenKMIP_pykmip_pr597",
        "PagerDuty_python-pagerduty_pr62",
        "Peter-Slump_python-keycloak-client_pr19",
        "Polyconseil_django-cid_pr68",
        "Pylons_pyramid_retry_pr20",
        "PythonCharmers_python-future_pr596",
        "ReactiveX_rxpy_pr726",
        "RobinNil_file_read_backwards_pr87",
        "Shoobx_xmldiff_pr142",
        "SoCo_soco_pr901",
        "Yubico_python-fido2_pr233",
        "devcycleHQ_python-server-sdk_pr91",
        "google_etils_pr726",
        "grumBit_aws_cron_expression_validator_pr19",
        "pymanopt_pymanopt_pr201",
        "pyro-ppl_pyro-api_pr3",
        "toxicphreAK_python-docx-ng_pr7",
    }
)

NL2REPO_EXCLUDED_TASK_IDS = frozenset({"pdfplumber-stable"})

_EXCLUDED_BY_PROFILE = {
    "repo_generation_denovoswe": DENOVOSWE_EXCLUDED_TASK_IDS,
    "repo_generation_nl2repo": NL2REPO_EXCLUDED_TASK_IDS,
}


def _task_profile(task: Any) -> str:
    metadata = task.metadata if isinstance(getattr(task, "metadata", None), dict) else {}
    rllm_metadata = metadata.get("rllm") or {}
    return str(
        metadata.get("task_profile")
        or rllm_metadata.get("task_profile")
        or ""
    )


def filter_inaccessible_repo_generation_tasks(
    tasks: Iterable[Any],
) -> tuple[list[Any], list[str]]:
    """Return schedulable tasks and sorted explicitly excluded instance IDs."""

    kept: list[Any] = []
    removed: list[str] = []
    for task in tasks:
        task_id = str(getattr(task, "id", ""))
        excluded = _EXCLUDED_BY_PROFILE.get(_task_profile(task), frozenset())
        if task_id in excluded:
            removed.append(task_id)
        else:
            kept.append(task)
    return kept, sorted(removed)


__all__ = [
    "DENOVOSWE_EXCLUDED_TASK_IDS",
    "NL2REPO_EXCLUDED_TASK_IDS",
    "filter_inaccessible_repo_generation_tasks",
]
