"""Local-harness prompt loader: the three enrichment system prompts
live in prompt-registry (INTERNAL categories, 2026-09-08); deployed
states receive them via the plan. Harnesses read the SIBLING REPO's
files through the registry's OWN parser (prompt_md — one grammar, one
parser; a private regex here once meant harness evals could run
different prompt bytes than production binds). Valid only on a
workspace checkout, which is the only place harnesses run."""
import os
import sys

_REG_REPO = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "prompt-registry-on-aws")
sys.path.insert(0, os.path.join(_REG_REPO, "gateway_tool"))
import prompt_md  # noqa: E402  (the registry's canonical parser)


def system(category: str) -> str:
    path = os.path.join(_REG_REPO, "prompts", category, "default.md")
    text = open(path, encoding="utf-8").read()
    return prompt_md.parse_prompt_md(text)["system"]
