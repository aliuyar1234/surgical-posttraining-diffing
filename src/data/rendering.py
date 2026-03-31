from __future__ import annotations

import hashlib


def compute_template_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def render_neutral_prefix(prompt: str, template: str) -> str:
    return template.format(prompt=prompt)


def render_full_sequence(prompt: str, completion: str, template: str) -> str:
    return render_neutral_prefix(prompt, template) + completion
