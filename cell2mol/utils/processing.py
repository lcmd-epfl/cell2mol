"""Shared failure classification and CLI exit-status handling."""

import logging
import sys

from cell2mol.utils import config
from cell2mol.utils.exceptions import ASEParseError
from cell2mol.utils.limits import ProcessingTimeoutError

logger = logging.getLogger(__name__)


class UnsupportedInput(Exception):
    """Input rejected before processing because it is not supported."""


class ProcessingError(Exception):
    """Processing failure with an exit code and any available partial result."""

    def __init__(self, message, exit_code, result=None):
        super().__init__(message)
        self.exit_code = exit_code
        self.result = result


def exception_exit_code(exc):
    """Classify input, ASE, memory, timeout, and other processing failures."""
    if isinstance(exc, ProcessingError):
        return exc.exit_code
    if isinstance(exc, UnsupportedInput):
        return config.ERR_INPUT
    if isinstance(exc, MemoryError):
        return config.ERR_MEMORY
    if isinstance(exc, ProcessingTimeoutError):
        return config.ERR_TIMEOUT
    if isinstance(exc, ASEParseError):
        return config.ERR_ASE_PARSE
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, PermissionError)):
        return config.ERR_INPUT
    return config.ERR_GENERAL


def read_structure(reader, input_path, file_format):
    """Wrap ASE parse failures while preserving resource and input errors."""
    try:
        return reader(input_path, format=file_format)
    except (MemoryError, ProcessingTimeoutError, OSError):
        raise
    except Exception as exc:
        raise ASEParseError(f"ASE failed to parse {input_path}: {exc}") from exc


def raise_processing_error(exc, result=None, *objects):
    """Record a runtime failure on partial models and raise its classified error."""
    code = exception_exit_code(exc)
    mode = {
        config.ERR_MEMORY: "memory",
        config.ERR_TIMEOUT: "timeout",
        config.ERR_GENERAL: "general",
        config.ERR_ASE_PARSE: "ase_parse",
        config.ERR_INPUT: "input",
    }.get(code, "general")
    for obj in objects or (result,):
        if obj is None:
            continue
        if hasattr(obj, "error_cases"):
            if obj.error_cases is None:
                obj.error_cases = {}
            if obj.error_cases_all is None:
                obj.error_cases_all = {}
            obj.error_cases[mode] = code
            obj.error_cases_all[mode] = [code]
        elif hasattr(obj, "error_case"):
            codes = list(obj.error_cases_all or [])
            if code not in codes:
                codes.append(code)
            obj.error_cases_all = codes
            obj.error_case = code
    if isinstance(exc, ProcessingError):
        if result is not None:
            exc.result = result
        raise exc
    raise ProcessingError(str(exc), code, result) from exc


def save_partial_results(save, result=None, *objects):
    """Attempt final output without replacing a pending processing failure."""
    pending = sys.exc_info()[1]
    try:
        save()
    except Exception as exc:
        logger.error("Final output failed: %s", exc)
        if pending is None:
            raise_processing_error(exc, result, *objects)


def result_exit_code(result):
    """Return zero on success, 50 on domain failure, or a recorded runtime code."""
    if result is None:
        return config.ERR_CELL2MOL
    if getattr(result, "type", None) == "cells":
        codes = [result_exit_code(result.reference)]
        codes.append(result_exit_code(result.unitcell))
        return next(
            (code for code in codes if code not in (0, config.ERR_CELL2MOL)),
            config.ERR_CELL2MOL if any(codes) else 0,
        )
    if hasattr(result, "error_cases"):
        codes = list((result.error_cases or {}).values())
    else:
        codes = list(result.error_cases_all or [])
        codes.append(result.error_case or 0)
    runtime_codes = {
        config.ERR_INPUT,
        config.ERR_ASE_PARSE,
        config.ERR_GENERAL,
        config.ERR_TIMEOUT,
        config.ERR_MEMORY,
    }
    return next(
        (code for code in codes if code in runtime_codes),
        config.ERR_CELL2MOL if any(codes) else 0,
    )


def run_cli(process, **kwargs):
    """Run processing at a CLI boundary and return its exit status."""
    try:
        return result_exit_code(process(**kwargs))
    except Exception as exc:
        code = exception_exit_code(exc)
        logger.error("Processing failed (exit code %d): %s", code, exc)
        return code
