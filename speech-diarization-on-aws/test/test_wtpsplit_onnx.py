#!/usr/bin/env python3
"""
test_wtpsplit_onnx.py
─────────────────────
Smoke-test the exported INT8 wtpsplit ONNX model by loading it directly
and running sentence splitting across 12 languages.

Usage
─────
    python test_wtpsplit_onnx.py --model-dir /path/to/model/dir

    The model directory must contain:
        model_optimized.onnx  ← INT8 quantized (produced by export_wtpsplit.py)
        config.json           ← model config (use_fp16=False)
        tokenizer.json        ← XLM-RoBERTa fast tokenizer

    To extract from the tar.gz archive produced by the exporter:
        tar -xzf wtpsplit_sat3lsm_int8.tar.gz -C /tmp/wtpsplit_model

Exit codes
──────────
    0  all tests passed
    1  one or more tests failed
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import argparse
import sys
import textwrap
import time


# ──────────────────────────────────────────────────────────────────────────────
# Test corpus — (language_code, display_name, input_text, expected_sentences)
# ──────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    (
        "en", "English",
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump!",
        3,
    ),
    (
        "fr", "French",
        "Le renard brun rapide saute par-dessus le chien paresseux. "
        "Portez ce vieux whisky au juge blond qui fume. "
        "Voici une troisième phrase pour faire bonne mesure.",
        3,
    ),
    (
        "de", "German",
        "Der schnelle braune Fuchs springt über den faulen Hund. "
        "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich. "
        "Das ist der dritte Satz.",
        3,
    ),
    (
        "es", "Spanish",
        "El veloz zorro marrón salta sobre el perro perezoso. "
        "Jovencillo emponzoñado de whisky, ¡qué figurota exhibe! "
        "Esta es la tercera oración.",
        3,
    ),
    (
        "pt", "Portuguese",
        "O rato roeu a roupa do rei de Roma. "
        "Zebras caolhas de Java querem passar fax para moças gigantes. "
        "Esta é a terceira frase.",
        3,
    ),
    (
        "nl", "Dutch",
        "De snelle bruine vos springt over de luie hond. "
        "Pa's wijze lynx bezag vroom het fikse aquaduct. "
        "Dit is de derde zin.",
        3,
    ),
    (
        "it", "Italian",
        "La volpe marrone veloce salta sopra il cane pigro. "
        "Pranzo d'acqua fa volti sghembi. "
        "Questa è la terza frase.",
        3,
    ),
    (
        "cy", "Welsh",
        "Dyma'r frawddeg gyntaf mewn Cymraeg. "
        "Mae'r ail frawddeg yn dilyn y cyntaf. "
        "A dyma'r drydedd frawddeg.",
        3,
    ),
    (
        "ar", "Arabic",
        "الثعلب البني السريع يقفز فوق الكلب الكسول. "
        "هذه هي الجملة الثانية باللغة العربية. "
        "وهذه هي الجملة الثالثة والأخيرة.",
        3,
    ),
    (
        "zh", "Chinese",
        "今天天气很好。我们去公园散步吧。外面阳光明媚，非常适合出行。",
        3,
    ),
    (
        "ja", "Japanese",
        "素早い茶色のキツネが怠け者の犬を飛び越えた。"
        "これは二番目の日本語の文です。"
        "そしてこれが三番目の文です。",
        3,
    ),
    (
        "ko", "Korean",
        "빠른 갈색 여우가 게으른 개를 뛰어넘었다. "
        "이것은 두 번째 한국어 문장입니다. "
        "그리고 이것이 세 번째 문장입니다.",
        3,
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_tests(model_dir: str, tolerance: int = 1) -> bool:
    # Uses lib.wtpsplit_sat.SaT — no wtpsplit package, no torch, no transformers.
    # fp32 attention_mask is handled correctly by default in our vendored SaT.
    from lib.wtpsplit_sat import SaT

    print(f"\n  Loading SaT from: {model_dir}")
    t0 = time.monotonic()
    model = SaT(model_dir, ort_providers=["CPUExecutionProvider"])
    load_ms = (time.monotonic() - t0) * 1000
    print(f"  Model loaded in {load_ms:.0f} ms\n")

    col_lang = 12
    col_got = 6
    col_exp = 6
    header = (
        f"  {'Language':<{col_lang}}  {'Got':>{col_got}}  {'Exp':>{col_exp}}  "
        f"{'ms':>6}  Result    Preview"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    passed = failed = 0

    for lang_code, lang_name, text, expected in TEST_CASES:
        t1 = time.monotonic()
        sentences = list(model.split(text))
        elapsed_ms = (time.monotonic() - t1) * 1000

        got = len(sentences)
        diff = abs(got - expected)
        ok = diff <= tolerance
        icon = "✅" if ok else "❌"

        # Show a short preview of the first sentence
        preview = sentences[0].strip() if sentences else "(empty)"
        preview = textwrap.shorten(preview, width=42, placeholder="…")

        line = (
            f"  {lang_name:<{col_lang}}  {got:>{col_got}}  {expected:>{col_exp}}  "
            + f"{elapsed_ms:>5.0f}ms  {icon}  \"{preview}\""
        )
        print(line)

        if ok:
            passed += 1
        else:
            failed += 1
            # Print full split so it's easy to debug
            for i, s in enumerate(sentences, 1):
                print(f"       [{i}] {s.strip()!r}")

    print()
    total = passed + failed
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  🎉")
    print()

    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the INT8 wtpsplit ONNX model across multiple languages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            ────────
            # Model dir unpacked from the archive:
            python test_wtpsplit_onnx.py --model-dir /tmp/wtpsplit_model

            # Looser tolerance (allow ±2 sentence count difference):
            python test_wtpsplit_onnx.py --model-dir /tmp/wtpsplit_model --tolerance 2
        """),
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing model.onnx and config.json",
    )
    parser.add_argument(
        "--tolerance", type=int, default=1,
        help="Max allowed sentence-count difference vs expected (default: 1)",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  SaT INT8 ONNX — sentence split smoke test (lib.wtpsplit_sat)")
    print("=" * 64)

    all_passed = run_tests(args.model_dir, tolerance=args.tolerance)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
