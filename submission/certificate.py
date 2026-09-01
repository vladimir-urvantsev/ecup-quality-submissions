"""Compatibility shim for the immutable soup6 submission entry point.

The old scorer imports ``certificate.build_comment``.  Candidate A packages
this file under that name, so all score/model/threshold bytes remain immutable
while only post-verdict prose is replaced by the audited production renderer.
"""

from production_reviewer_rationale import render_comment


def build_comment(
    category: str,
    name: str,
    description: str,
    label: int,
    scored: bool = True,
) -> str:
    return render_comment(
        category, name, description, int(label), scored=bool(scored)
    )


__all__ = ["build_comment"]
