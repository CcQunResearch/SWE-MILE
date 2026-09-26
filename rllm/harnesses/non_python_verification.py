"""Private normalized SWE-rebench shadow evidence (stdlib only).

This module is bundled into the shadow parser, never the official grader.
Display aliases are accepted only when one observed owner and one contract
node agree. Unknown identities and conflicting states remain unavailable.
"""

import json
import os
import re

NON_PYTHON_EVIDENCE_VERSION = 2
_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_DURATION = re.compile(r"\s*\(\d+(?:\.\d+)?\s*(?:ms|s)\)\s*$")


def _title(value):
    return _DURATION.sub("", value.strip())


def _hierarchy(value):
    # Only actual display separators, never '<div>', '>0', or regex syntax.
    return re.sub(r"\s+(?:›|>)\s+", " ", _title(value))


class EvidenceError(ValueError):
    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.evidence = evidence


def _mocha_list_records(clean):
    """List prints full titles, including literal newlines in template tests.

    The official leaf contract retains the trailing colon and truncates at a
    newline. Preserve both the full owner and that display key so a truncated
    mixed-state contract remains ambiguous instead of silently becoming PASS.
    """
    lines = clean.splitlines()
    records = []
    details = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\d+ (?:passing|failing|pending)\b", stripped):
            details = True
            continue
        match = re.match(r"^[✓✔]\s+(.+)$", stripped) if not details else re.match(r"^\d+\) (.+)$", stripped)
        if not match:
            continue
        parts = [match[1]]
        end = index + 1
        if details:
            while not parts[-1].endswith(":") and end < len(lines) and lines[end].strip():
                parts.append(lines[end].strip())
                end += 1
            if not parts[-1].endswith(":"):
                continue
            status = "FAILED"
        else:
            while not re.search(r": \d+(?:\.\d+)?ms$", parts[-1]) and end < len(lines):
                if re.match(r"^\s*(?:[✓✔]|\d+\)) ", lines[end]):
                    break
                parts.append(lines[end].strip())
                end += 1
            if not re.search(r": \d+(?:\.\d+)?ms$", parts[-1]):
                continue
            parts[-1] = re.sub(r" \d+(?:\.\d+)?ms$", "", parts[-1])
            status = "PASSED"
        full = "\n".join(parts)
        records.append((full, parts[0], status, full))
    return records


def _named_records(log, language):
    """Return (full title, leaf title, status, owner) from runner records.

    Failure detail blocks are parsed as blocks; assertion diffs and stack
    traces are not test declarations. Ownership includes the test file when
    the runner prints it and the complete suite title in all cases.
    """
    clean = _ANSI.sub("", log).replace("\r", "\n")
    records = []
    if language == "go":
        pending = []
        for line in clean.splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                event = None
            if isinstance(event, dict) and event.get("Test"):
                status = {"pass": "PASSED", "fail": "FAILED", "skip": "SKIPPED"}.get(event.get("Action"))
                if status:
                    name = event["Test"]
                    records.append((name, name, status, str(event.get("Package", "")) + "::" + name))
                continue
            match = re.match(r"^\s*--- (PASS|FAIL|SKIP):\s+(.+?)\s+\([\d.]+s\)", line)
            if match:
                pending.append((match[2], {"PASS": "PASSED", "FAIL": "FAILED", "SKIP": "SKIPPED"}[match[1]]))
            package = re.match(r"^(?:ok|FAIL)\s+(\S+)(?:\s|$)", line)
            if package:
                records.extend((name, name, status, package[1] + "::" + name) for name, status in pending)
                pending = []
        # An unterminated package has no trustworthy package ownership.
        return records

    is_jest = bool(re.search(r"Test Suites:|^\s*● ", clean, re.M))
    is_ava = bool(re.search(r"\[fail\]:|^\s*\d+ tests? (?:failed|passed)", clean, re.M)) and not is_jest
    is_mocha = bool(re.search(r"^\s*\d+ (?:passing|failing)\b", clean, re.M)) and not (is_jest or is_ava)
    if is_mocha and re.search(r"^\s*[✓✔] .+: \d+(?:\.\d+)?ms$", clean, re.M):
        return _mocha_list_records(clean)
    suites = []
    file_name = ""
    lines = clean.splitlines()
    in_details = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if is_jest:
            file_match = re.match(r"^(?:PASS|FAIL)\s+(\S+\.[cm]?[jt]sx?)(?:\s|$)", stripped)
            if file_match:
                file_name = file_match[1]
                suites = []
                in_details = False
                continue
            detail = re.match(r"^● (.+)$", stripped)
            if detail:
                in_details = True
                full = _title(detail[1])
                if full == "Test suite failed to run":
                    continue
                leaf = full.rsplit(" › ", 1)[-1]
                records.append((full, leaf, "FAILED", file_name + "::" + _hierarchy(full)))
                continue
            if in_details:
                continue
            test = re.match(r"^([✓✔✕×✖○])\s+(?:skipped\s+)?(.+)$", stripped)
            if test:
                leaf = _title(test[2])
                parent = [name for depth, name in suites if depth < indent]
                full = " › ".join([*parent, leaf])
                status = "PASSED" if test[1] in "✓✔" else "SKIPPED" if test[1] == "○" else "FAILED"
                records.append((full, leaf, status, file_name + "::" + _hierarchy(full)))
            elif stripped and indent >= 2 and not re.match(r"(?:Test Suites:|Tests:|Snapshots:|Time:|Ran all|console\.|at |\d+\s*\|)", stripped):
                suites = [(depth, name) for depth, name in suites if depth < indent]
                suites.append((indent, stripped))
        elif is_ava:
            test = re.match(r"^([✔✓✘✖×])\s+(?:\[(fail|skip)\]:?\s*)?(.+)$", stripped)
            if test and "␊" not in stripped:
                full = _title(test[3])
                status = "SKIPPED" if test[2] == "skip" else "FAILED" if test[1] in "✘✖×" else "PASSED"
                records.append((full, full.rsplit(" › ", 1)[-1], status, full))
        elif is_mocha:
            if re.match(r"^\d+ (?:passing|failing|pending)\b", stripped):
                in_details = True
                continue
            header = re.match(r"^\d+\) (.+)$", stripped)
            if in_details:
                if not header:
                    continue
                parts = [header[1]]
                # Mocha failure titles occupy the contiguous, increasingly
                # indented lines before a blank line / terminal colon.
                for following in lines[index + 1 :]:
                    if parts[-1].endswith(":") or not following.strip():
                        break
                    if len(following) - len(following.lstrip()) <= indent:
                        break
                    parts.append(following.strip())
                if parts[-1].endswith(":"):
                    parts[-1] = parts[-1][:-1]
                    records.append((" ".join(parts), parts[-1], "FAILED", " ".join(parts)))
                continue
            test = re.match(r"^(?:[✔✓]|(\d+)\))\s+(.+)$", stripped)
            skipped = re.match(r"^- (.+)$", stripped)
            if test or skipped:
                leaf = _title(test[2] if test else skipped[1])
                parent = [name for depth, name in suites if depth < indent]
                full = " ".join([*parent, leaf])
                status = "FAILED" if test and test[1] else "PASSED" if test else "SKIPPED"
                records.append((full, leaf, status, full))
            elif stripped and indent >= 2:
                suites = [(depth, name) for depth, name in suites if depth < indent]
                suites.append((indent, stripped))
    return records


def reconcile_named_results(results, log, expected, language):
    records = _named_records(log, language)
    expected = set(expected)
    candidates = {}
    unknown = set()
    normalized_expected = {}
    for name in expected:
        normalized_expected.setdefault(_hierarchy(name), []).append(name)
    known_record_names = {name for full, leaf, _, _ in records for name in (full, leaf)}
    # The official parsers use leaf keys in several benchmark contracts.
    # Corroborating such an existing key is not a new alias assignment.
    # Only new aliases and partition identities require a unique owner.
    for full, leaf, status, owner in records:
        aliases = list({name for name in (full, owner) if name in expected})
        if not aliases:
            aliases = normalized_expected.get(_hierarchy(full), [])
        if not aliases:
            aliases = [leaf] if leaf in expected else []
        if len(aliases) > 1:
            raise EvidenceError("ambiguous_test_ownership", mapping_candidates=sorted(aliases)[:20], mapping_title=full)
        if not aliases:
            unknown.update((full, leaf))
            continue
        name = aliases[0]
        canonical_owner = _hierarchy(owner)
        owners = candidates.setdefault(name, {})
        owners.setdefault(canonical_owner, set()).add(status)
    repaired = dict(results)
    unique_owners = {}
    corroborated = []
    for name, owners in candidates.items():
        statuses = set().union(*owners.values())
        evidence = {"mapping_test": name, "mapping_owners": sorted(owners)[:20], "mapping_states": sorted(statuses), "official_status": results.get(name)}
        if len(owners) > 1:
            if name in results and statuses == {results[name]}:
                corroborated.append(name)
                # Do not export an invented owner for partition selectors.
                continue
            raise EvidenceError("ambiguous_test_ownership", **evidence)
        if len(statuses) != 1 or (name in results and results[name] not in statuses):
            raise EvidenceError("test_state_conflict", **evidence)
        status = next(iter(statuses))
        unique_owners[name] = next(iter(owners))
        # Supplement failure observations only. A new text-only success is
        # insufficient to create positive reward; it still needs the official
        # vector or a separately bound native collector.
        if status != "PASSED" or name in results:
            repaired[name] = status
    # Drop only proven non-target runner records and recognizable diagnostic
    # noise; retain unknown extras so host contract-fill stays fail-closed.
    ignored = []
    for name in list(repaired):
        if name in expected:
            continue
        diagnostic = any(token in name for token in ("␊", "AssertionError", "assert(false)", "at ---"))
        known = name in unknown or name in known_record_names
        if diagnostic or known:
            ignored.append(name)
            del repaired[name]
    return repaired, {
        "adapter_version": NON_PYTHON_EVIDENCE_VERSION,
        "named_records": len(records),
        "recovered_tests": len(set(repaired) - set(results)),
        "known_non_target_count": len(ignored),
        "known_non_target_sample": sorted(ignored)[:20],
        "test_owners": unique_owners,
        "corroborated_official_key_count": len(corroborated),
        "corroborated_official_key_sample": sorted(corroborated)[:20],
    }


def execution_failure(log, execution, *, complete=False):
    """Hard failures also block the older F2P-only contract-fill fallback."""
    clean = _ANSI.sub("", log)
    commands = execution.get("commands") or []
    if execution.get("timed_out") or any(c.get("timed_out") or c.get("exit_code") in (124, 143) for c in commands):
        return "verifier_timeout"
    if execution.get("resource_exhausted") or any(c.get("oom_kill_delta", 0) or c.get("exit_code") == 137 for c in commands):
        return "resource_exhausted"
    if any(c.get("signal") or c.get("exit_code") in (134, 139) for c in commands):
        return "verifier_crash"
    failed = any(c.get("exit_code", 0) != 0 for c in commands)
    if failed and re.search(r"out of memory|cannot allocate memory|no space left", clean, re.I):
        return "resource_exhausted"
    # Error codes are common test titles (and assertion operands). They do
    # not prove an interrupted dependency operation. Keep standalone runner
    # diagnostics fail-closed, and never mask explicit package-manager errors.
    diagnostics = "\n".join(
        line for line in clean.splitlines()
        if not re.match(r"^\s*(?:[✓✔✕×✖○✘]|●|--- (?:PASS|FAIL|SKIP):|=== (?:RUN|PAUSE|CONT)|\d+\))\s*", line)
        and not re.match(r"^\s*(?:[+\-]|(?:expect|assert)[.(]|\d+\s*\|)", line)
    )
    network_pattern = (
        r"ENOTFOUND|EAI_AGAIN|ECONN\w+|ETIMEDOUT|no such host|network is unreachable|TLS handshake|"
        r"failed to (?:fetch|download)|Could not resolve|Temporary failure in name resolution"
    )
    if failed and re.search(network_pattern, diagnostics, re.I):
        dependency_operation = re.search(
            r"(?mi)^(?:npm (?:ERR!|error)\s+.*(?:network|fetch|request|EAI_AGAIN|ENOTFOUND|ECONN|ETIMEDOUT)|"
            r"go: .*https?://|[^\n]*\.go:\d+:\d+: .*https?://|"
            r"(?:error|fatal): .*https?://|.*(?:failed to (?:fetch|download)|Could not resolve host)\b)",
            diagnostics,
        )
        if dependency_operation or not complete:
            return "dependency_network"
    return None


def source_abort(log, language, execution):
    """Positive source diagnostics plus failed execution; never just exit!=0."""
    clean = _ANSI.sub("", log)
    commands = execution.get("commands") or []
    if execution_failure(log, execution) or not any(c.get("exit_code", 0) != 0 for c in commands):
        return None
    if language == "go":
        pattern = (
            r"(?m)^([^\s:]+\.go):\d+(?::\d+)?:\s*(?:undefined:|cannot use |.*redeclared|"
            r".*already declared|.*undefined|.*(?:imported|declared)(?: and)? (?:but )?not used|syntax error|too many arguments|"
            r"not enough arguments|invalid operation:|cannot convert |cannot assign to |missing return|"
            r".*has no field or method|.*does not implement)"
        )
        paths = sorted(set(re.findall(pattern, clean)))
        if paths and "[build failed]" in clean:
            return {"phase": "source_compile", "paths": paths}
    else:
        paths = sorted(set(re.findall(
            r"(?m)([^\s:()]+\.[cm]?[jt]sx?)(?::\d+:\d+|\(\d+,\d+\):)\s*(?:-\s*)?(?:error\s+)?TS\d+:", clean
        )))
        # tsc/ts-node (Mocha) use file(line,column), while ts-jest uses
        # file:line:column. Both carry explicit TypeScript diagnostic codes.
        if paths:
            return {"phase": "source_compile", "paths": paths}
        paths = sorted(set(re.findall(r"(?m)^\s*(/[^\s:]+\.[cm]?[jt]sx?):\d+", clean)))
        if paths and re.search(r"SyntaxError:|ReferenceError:", clean) and re.search(r"Test suite failed to run|Unexpected token|Unexpected identifier", clean):
            return {"phase": "source_load", "paths": paths}
    return None


def verify_probe_binding(contract, token, results_path, output_path):
    """The serial shadow run deletes old outputs before writing its token."""
    path = contract["probe_binding_path"]
    with open(path, encoding="utf-8") as handle:
        observed = handle.read()
    if not token or observed != token:
        raise ValueError("non_python_probe_binding_mismatch")
    started = os.stat(path).st_mtime_ns
    if any(os.stat(candidate).st_mtime_ns < started for candidate in (results_path, output_path)):
        raise ValueError("non_python_stale_results")


def canonicalize_partition_artifact(artifact, expected, owners):
    """Map native full titles using previously observed exact owner identities.

    Do not infer membership from suffixes or repair aggregate totals. Once
    names are mapped, the caller calibrates the explicit per-test vector.
    """
    if not isinstance(artifact, dict) or not isinstance(artifact.get("test_results"), dict):
        return artifact
    aliases = {}
    for name, owner in owners.items():
        for alias in (name, owner, owner.split("::", 1)[-1]):
            aliases.setdefault(_hierarchy(alias), set()).add(name)
    results = {}
    for raw, status in artifact["test_results"].items():
        candidates = aliases.get(_hierarchy(raw), set())
        if len(candidates) > 1:
            raise ValueError("ambiguous_test_ownership")
        name = next(iter(candidates)) if candidates else raw
        if candidates and name not in expected and status == "SKIPPED":
            # Jest includes explicitly unselected tests as pending. Only
            # exclude those whose non-membership is proven by owner identity.
            continue
        if name in results:
            raise ValueError("duplicate_partition_test_identity")
        results[name] = status
    repaired = dict(artifact)
    repaired["test_results"] = results
    repaired["diagnostics"] = dict(artifact.get("diagnostics") or {})
    repaired["diagnostics"].pop("partition_observation", None)
    return repaired


def parser_script(legacy_script):
    """Generate a separate parser bundle; the Python bundle stays byte exact."""
    with open(__file__, encoding="utf-8") as handle:
        source = handle.read()
    hook = """
import hashlib
non_python_evidence = {}
try:
    verify_probe_binding(payload["non_python_evidence"], sys.argv[2], results_path, output_path)
    early_failure = execution_failure(clean_log, artifact_execution, complete=True)
    if early_failure:
        raise EvidenceError(early_failure, execution=artifact_execution)
    if not (resource_exhausted or toolchain_unavailable or diagnostic_timeout or unsafe_crash):
        if len(command_executions) == 1:
            results, non_python_evidence = reconcile_named_results(results, clean_log, expected_test_order, expected_language)
            non_python_evidence["command_index"] = command_executions[0]["index"]
            non_python_evidence["source_abort"] = source_abort(clean_log, expected_language, artifact_execution)
        else:
            # Preserve already usable official vectors. No new text aliases
            # or abort completion without a unique command owner.
            non_python_evidence = {"adapter_version": NON_PYTHON_EVIDENCE_VERSION, "mapping_failure": "command_ownership_unavailable"}
        non_python_evidence["execution"] = artifact_execution
        non_python_evidence["probe_token_sha256"] = hashlib.sha256(sys.argv[2].encode()).hexdigest()
    observed_expected_tests = set(results) & expected_tests
    hard_failure = execution_failure(clean_log, artifact_execution, complete=expected_tests <= set(results))
    if hard_failure:
        raise EvidenceError(hard_failure, execution=artifact_execution)
    # Override only this private bundle's broad legacy text heuristic.
    dependency_network_error = False
except (ValueError, OSError, KeyError, IndexError) as exc:
    evidence = {"adapter_version": NON_PYTHON_EVIDENCE_VERSION, "execution": artifact_execution,
                "missing_count": len(expected_tests - set(results)), "observed_count": len(expected_tests & set(results))}
    evidence.update(getattr(exc, "evidence", {}))
    print(json.dumps({"ok": False, "failure_type": str(exc), "error": str(exc), "failure_stage": "non_python_evidence",
                      "failure_origin": "shadow_parser", "failure_evidence": evidence, "log_tail": log[-tail_chars:]}))
    raise SystemExit(0)
"""
    script = legacy_script.replace("def emit_results():", hook + "\ndef emit_results():", 1)
    script = script.replace('"observed_expected_count": len(observed_expected_tests),', '"observed_expected_count": len(observed_expected_tests), "non_python_evidence": non_python_evidence,')
    return source + "\n" + script
