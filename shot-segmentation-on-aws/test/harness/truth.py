"""Ground truth for the Phase 9 corpus — the ONLY copy. Every replay imports it.

Three label sets, all eyeballed from the per-shot debug grids and the
`sheet_all.py` contact sheets:

  TRUE_CARDS     the six verified genuine structural cards, as exact shot-id
                 sets. These are what a correct gate stack must keep.
  CULLED_LABELS  FURN / SET / PROMO for the 25 groups the pre-swap 9.3 ceiling
                 deleted (historical — keyed to pre-swap member counts).
  AUDIT_LABELS   the 2026-08-21 published-pool audit, keyed by exact shot-id
                 set: every group over the reference ceiling or carrying the
                 set signature (53), PLUS the kept-pool sampling audit (53 of
                 the remaining 356 — AUDIT_SAMPLE_LABELS, folded in).

Memberships are shot-id sets, never `(run, group_id)` pairs. group_id is an
artefact of whatever clustering produced it — the moment a replay changes a
threshold the published ids stop lining up and identity by id becomes silently
wrong (it made the drama_e card read as LOST at t=0.07 when it had in fact survived
intact). Shot ids are stable across every replay of one asset. There is
deliberately no `(run, group_id)` accessor here: a second keying of the same
truth is a second thing to keep in sync.

This file previously existed in seven places — `verdict.py` held CULLED_LABELS
(as `LAB`) plus its own copy of the cards, `guardrails.py` and `rule_test.py`
`exec()`'d verdict.py's source by string-splitting to borrow them, and
`chaining.py`, `match_graph.py` and `separate.py` each carried a third form
keyed by `(run, group_id)`. Two copies of a label set can disagree without
either looking wrong, so they are consolidated here.
"""

# --- the six verified genuine cards -----------------------------------------
# Keys are "<run>:g<published group_id at the shipped 0.05>" for readability
# only; nothing resolves them back to an id.
TRUE_CARDS = {
    "drama_e:g0":       {"name": "drama_e",         # act-break card, 4 occ
                          "shots": [148, 270, 447, 674]},
    "drama_a:g0":      {"name": "drama_a",        # "<CHANNEL> <digit>" card, 5
                          "shots": [324, 450, 628, 862, 1035]},
    "markets_a:g0": {"name": "markets_a",   # programme ident, 3
                          "shots": [1, 153, 281]},
    "markets_a:g3": {"name": "markets_a",   # second ident variant, 3
                          "shots": [4, 156, 284]},
    "reality_a:g0":      {"name": "reality_a",        # title card, 8 —
                          # the asset that pins 9.3's ceiling, surviving at
                          # exactly the limit
                          "shots": [2, 290, 538, 668, 781, 937, 1066, 1112]},
    "title_card_a:g0":       {"name": "title_card_a",         # title card, 3
                          "shots": [29, 212, 468]},
}

# --- verdicts on the 25 groups the shipped ceiling culls ---------------------
# FURN  genuine structural/branded furniture the ceiling destroys (a false cull)
# SET   recurring location or camera the ceiling correctly kills
# PROMO the same promo/package re-airing — genuine recurrence, but not a card
#
# Keyed by (run, raw member count), which is unique within a run except for
# news_a n=17, where two distinct groups collide; that one entry maps
# occurrence count -> label. Callers must go through `label_culled`. The
# previous copies disambiguated the collision with a magic `occ_ratio > 0.45`
# in three files and `occ == 8` in a fourth — the same test by coincidence, and
# a fitted threshold standing in for a fact about two known groups.
CULLED_LABELS = {
    ("markets_a", 11): "FURN",   # markets_a programme title card
    ("markets_a", 13): "FURN",   # blue gradient transition wipe
    ("news_b", 24): "FURN",          # network animated ID
    ("news_b", 18): "FURN",          # streamer logo bumper
    ("news_a", 21): "FURN",          # <NETWORK> PSA end card
    ("news_a", 17): {8: "FURN",      # "ONLY ON <STREAMER>" end card
                      7: "PROMO"},    # PSA star set, presenter varies
    # Corrected 2026-08-21 during the published-pool audit: this is the
    # ad-break countdown slate (teaser plate + "WILL RETURN IN N:NN"), not a
    # re-aired package — the pre-swap audit made the same mid-frame mistake.
    ("news_b", 9): "FURN",
    ("news_a", 14): "PROMO",         # streamer show promo scene
    ("markets_a", 52): "SET",    # anchor desk — the corpus's largest
    ("markets_a", 26): "SET",
    ("markets_a", 20): "SET",
    ("markets_a", 12): "SET",
    ("news_b", 34): "SET",
    ("reality_c", 21): "SET",
    ("reality_c", 20): "SET",
    ("reality_c", 13): "SET",
    ("reality_c", 12): "SET",
    ("reality_a", 19): "SET",
    ("reality_a", 13): "SET",
    ("reality_a", 9): "SET",
    ("sports_doc_a", 47): "SET",
    ("sports_doc_a", 32): "SET",
    ("sports_doc_a", 27): "SET",
    ("sports_doc_a", 9): "SET",
}

# Only these two batches were labelled; a cull from any other asset is unknown,
# not mislabelled. Callers distinguish the two so an unlabelled cull can never
# be silently counted as a correct one.
LABELLED_BATCHES = ("phase9-batch", "paytv-batch")


def cards_for(name):
    """(key, set-of-shot-ids) for every true card belonging to one run."""
    return [(k, set(v["shots"])) for k, v in TRUE_CARDS.items()
            if v["name"] == name]


def is_true_card(name, shot_ids):
    """Does this exact membership match a verified card? Identity by shot ids,
    so it holds at any threshold."""
    want = set(shot_ids)
    return any(want == s for _, s in cards_for(name))


def label_culled(name, n, occ):
    """FURN / SET / PROMO for a culled group, or None if it was never labelled.

    `n` is the raw member count, `occ` the temporally separated occurrence
    count (needed only to break the news_a n=17 collision).
    """
    v = CULLED_LABELS.get((name, n))
    return v.get(occ) if isinstance(v, dict) else v


# --- verdicts on the 2026-08-21 published-pool visual audit ------------------
# The pre-swap audit covered only what the ceiling deleted; once 9.3 stopped
# dropping (annotate-not-drop), the PUBLISHED pool needed eyes. Audited strata:
# every published group at production defaults that is over the reference
# ceiling (recurrence_count > 8) OR carries the set signature
# (coverage > 0.016 AND ent_range < 1.0) — 53 groups, 691 member shots,
# enumerate_audit.py -> fetch_frames.sh -> sheet_all.py contact sheets.
#
# Labels: FURN / SET / PROMO as above, plus
#   TMPL  a recurring branded GRAPHIC TEMPLATE re-composited per airing
#         (market-chart panels, "DATA CHECK", a weather map). Not a place,
#         not the same render — a fourth shape the three old labels forced
#         into SET or PROMO by fiat.
#
# Findings the filter guidance rests on (as corrected by the user's review of
# the sheets, which caught three FURN the first pass had called PROMO):
#   * 9 FURN / 40 SET / 1 PROMO / 3 TMPL. Furniture splits into TWO classes
#     with opposite coverage behaviour:
#       - cards / idents / wipes (6): markets_a wipe + title card, network ident
#         + logo card, streamer end cards, network PSA end card — all at
#         coverage <= 0.0066, under half the 0.016 guidance threshold;
#       - ad-BREAK COUNTDOWN SLATES (3, all news_b): a designed teaser plate
#         with "<NETWORK> WILL RETURN IN N:NN" ticking bottom-right, one
#         per commercial break. Coverage 0.018-0.021 — set-like, NOT low —
#         because each break contributes ~15 s of slate. The most directly
#         consumer-valuable furniture in the corpus (it timestamps ad breaks)
#         and the class both audits initially mislabelled as re-aired PROMO
#         packages, because a mid-frame thumbnail shows the teaser footage
#         and the countdown is only legible zoomed.
#   * The set signature (coverage > 0.016 AND ent_range < 1.0) caught ONE
#     furniture group: the AI-election break slate at ent_range 0.99, a hair
#     under the 1.0 line its two siblings clear (1.99, 2.11). No scalar rule
#     separates the break-slate class from sets — it needs either the
#     countdown/plate itself (OCR or layout) or a human/VLM look.
#   * The set signature is one-directional regardless: 17 of the 40 SETs sit
#     at coverage <= 0.016 (interview shows slice one talking head thin), so
#     low coverage must never be read alone as "furniture".
#
# Keyed by exact shot-id set (the TRUE_CARDS principle) — (run, n) collides
# here: news_b has two n=9/occ=9 groups with DIFFERENT labels (anchor desk
# SET vs break countdown slate FURN).
AUDIT_LABELS = [
    ("news_b", "SET",  # two-anchor desk, chyron varies
     {0, 388, 409, 508, 747, 761, 782, 919, 1050}),
    ("news_b", "FURN",  # network logo card
     {97, 104, 191, 200, 318, 323, 472, 477, 543, 556, 649, 794, 805, 855, 862, 994}),
    ("news_b", "FURN",  # ad-BREAK COUNTDOWN SLATE, Nebraska-story teaser plate —
    # "<NETWORK> WILL RETURN IN N:NN" ticks bottom-right on every member
     {98, 192, 321, 475, 546, 653, 799, 998, 1114, 1256, 1358}),
    ("news_b", "FURN",  # network animated ident, sampled mid-animation
     {102, 197, 320, 327, 474, 482, 545, 553, 652, 798, 803, 860, 997, 1113, 1250, 1255, 1352, 1357}),
    ("news_b", "FURN",  # ad-BREAK COUNTDOWN SLATE, AI-election teaser plate —
    # first labelled PROMO from mid-frame thumbnails; the countdown clock is
    # only visible zoomed. The user's audit review caught it
     {319, 473, 544, 651, 797, 996, 1112, 1254, 1356}),
    ("news_b", "FURN",  # ad-BREAK COUNTDOWN SLATE, FBI-story teaser plate
     {324, 479, 549, 655, 801, 858, 1000, 1117, 1259, 1360}),
    ("news_a", "TMPL",  # TODAY'S HIGHS weather map
     {173, 627, 1389, 1767}),
    ("news_a", "PROMO",  # streamer show promo scene re-aired
     {342, 344, 443, 445, 880, 882, 981, 983, 1572, 1574, 1675, 1677, 2007, 2009}),
    ("news_a", "FURN",  # streamer end cards — TWO render variants merged (ONLY ON / STREAMING EXCLUSIVELY ON <STREAMER>), the pre-swap n=17 collision pair
     {347, 379, 448, 480, 885, 917, 986, 1021, 1028, 1577, 1610, 1680, 1715, 1942, 2012, 2047, 2054}),
    ("news_a", "FURN",  # <NETWORK> PSA END-CARD IDENT — every occurrence
    # immediately follows a PSA presenter shot on the star set, so it closes
    # a spot rather than standing alone as a transition
     {395, 397, 400, 417, 933, 935, 938, 955, 1044, 1629, 1632, 1649, 1958, 1960, 1963, 2070}),
    ("markets_a", "SET",
     {10, 271, 355, 358, 406}),
    ("markets_a", "SET",  # anchor desk — the corpus's largest tight cluster
     {16, 29, 48, 61, 65, 76, 83, 85, 87, 101, 103, 107, 135, 157, 170, 217, 247, 251, 269, 273, 336, 348, 351, 356, 359, 361, 374, 377, 405}),
    ("markets_a", "SET",
     {31, 63, 82, 84, 102, 136, 140, 143}),
    ("markets_a", "SET",
     {37, 188, 190, 375}),
    ("markets_a", "FURN",  # markets_a programme title card
     {43, 71, 111, 148, 197, 254, 277, 314, 340, 413}),
    ("markets_a", "FURN",  # blue gradient transition wipe
     {53, 66, 89, 104, 161, 192, 222, 262, 288, 306, 309, 388, 410}),
    ("markets_a", "TMPL",  # "DATA CHECK" segment data panel
     {77, 171, 244, 346, 390}),
    ("markets_a", "SET",
     {106, 248, 307, 376}),
    ("markets_a", "SET",
     {127, 139, 181, 218, 357, 368}),
    ("markets_a", "TMPL",  # full-screen market-chart panels, data differs
     {211, 215, 238, 242, 349, 392, 396, 400, 403}),
    ("reality_c", "SET",
     {103, 105, 273, 276, 350, 352, 355, 357, 483, 489}),
    ("reality_c", "SET",
     {123, 126, 139, 155, 158, 160, 246, 248, 250, 253, 513, 516, 537, 539, 541, 580, 582, 585, 589, 591}),
    ("sports_doc_a", "SET",
     {74, 655, 704, 716, 775, 781, 783, 1007, 1219, 1230, 1245, 1251, 1253}),
    ("sports_doc_a", "SET",
     {467, 498, 536, 882, 1131, 1137, 1155, 1178, 1293}),
    ("sports_doc_a", "SET",
     {506, 529, 845, 858, 862, 898, 1163, 1175, 1268, 1274, 1278, 1304}),
    ("studio_b1", "SET",
     {45, 47, 54, 112, 119, 133, 150, 158, 175, 183, 256, 285, 451, 508}),
    ("studio_b2", "SET",
     {55, 62, 136, 169, 171, 421, 452, 575, 620, 679, 719}),
    ("studio_b3", "SET",
     {165, 231, 267, 331, 394, 477, 554, 653, 666}),
    ("quiz_a", "SET",
     {23, 26, 28, 30, 32, 229, 359, 494, 499, 501, 614, 623, 714, 818}),
    ("quiz_a", "SET",
     {35, 82, 172, 221, 358, 439, 493, 612, 675, 705, 774, 808}),
    ("quiz_a", "SET",  # 52-member overhead set wide
     {36, 40, 46, 50, 58, 62, 73, 173, 182, 186, 194, 199, 212, 216, 321, 327, 337, 344, 352, 356, 440, 444, 452, 454, 459, 463, 467, 472, 476, 484, 489, 569, 574, 576, 580, 584, 591, 604, 608, 676, 678, 682, 687, 693, 697, 703, 775, 779, 787, 792, 799, 805}),
    ("quiz_a", "SET",
     {38, 71, 175, 177, 188, 204, 323, 329, 346, 348, 469, 474, 596}),
    ("quiz_a", "SET",  # podium framing that merges TWO contestants (Dylan+Colin)
     {47, 49, 70, 185, 187, 189, 191, 213, 215, 332, 349}),
    ("quiz_a", "SET",
     {108, 113, 234, 244, 283, 379, 617, 619, 621, 629, 631, 636, 716, 725, 727, 813}),
    ("quiz_a", "SET",
     {128, 271, 285, 390, 398, 412, 739, 745, 753}),
    ("quiz_a", "SET",
     {164, 169, 311, 313, 316, 429, 431, 436, 504, 555, 557, 562, 666, 672, 767, 769, 771, 811, 816, 820, 822, 826}),
    ("quiz_a", "SET",
     {383, 522, 524, 527, 529, 531, 533, 731, 740, 744, 750, 752, 755, 757}),
    ("quiz_a", "SET",
     {448, 450, 468, 470, 488, 566, 568, 581, 583, 595, 597, 610, 677, 684, 686, 694, 696, 704}),
    ("quiz_a", "SET",
     {451, 453, 471, 473, 475, 570, 572, 585, 587, 679, 681, 688, 690, 698, 700, 780, 782, 793, 795, 802}),
    ("quiz_a", "SET",
     {455, 457, 590, 776, 778, 785, 791, 797, 807}),
    ("factual_c", "SET",
     {440, 443, 465, 467, 502, 529, 548, 588, 595, 622}),
    ("factual_c", "SET",
     {450, 464, 468, 478, 480, 484, 486, 488, 490, 492, 495, 497, 499, 507, 509, 511, 522, 524, 526, 528, 530, 537, 545, 549, 552, 555, 559, 561, 565, 587, 589, 592, 594, 640, 642}),
    ("cooking_a", "SET",
     {168, 380, 386, 397, 464}),
    ("cooking_a", "SET",
     {342, 395, 448}),
    ("arts_doc_a", "SET",
     {55, 58, 82, 88, 129, 131}),
    ("arts_doc_a", "SET",
     {122, 124, 190, 192, 222, 297, 299, 337, 375, 410}),
    ("arts_doc_a", "SET",
     {141, 204, 206, 239, 269, 401, 406, 413}),
    ("arts_doc_a", "SET",
     {157, 339, 341, 427, 429}),
    ("docu_a", "SET",
     {12, 14, 68, 70, 79, 81, 156, 159, 276, 279, 445, 448, 1017}),
    ("docu_a", "SET",
     {32, 39, 457, 460, 472, 478, 501, 520, 529, 532, 535, 554, 557, 582, 628, 647, 655, 662, 665, 669, 748, 792, 795, 801, 804, 807, 810, 825, 828, 862}),
    ("docu_a", "SET",
     {72, 74, 104, 112, 125, 166, 222, 228, 231, 273, 929, 932, 962, 983, 986, 1011, 1014}),
    ("docu_a", "SET",
     {290, 304, 331, 376, 399, 428, 600, 607, 775, 790}),
    ("docu_a", "SET",
     {343, 362, 381, 482, 497, 524, 539, 592, 691, 762, 764, 768, 949, 957}),
]


def label_audited(name, shot_ids):
    """FURN / SET / PROMO / TMPL for an audited published group, or None.
    Identity by exact shot-id set, so it holds at any threshold."""
    want = set(shot_ids)
    for run, lab, ids in AUDIT_LABELS:
        if run == name and want == ids:
            return lab
    return None


# --- the 2026-08-21 KEPT-POOL SAMPLING AUDIT ---------------------------------
# The strata audit above covered over-ceiling and set-signature groups; this is
# the false-negative check on everything else: 53 groups sampled (seeded
# shuffle, every asset represented, the six verified true cards excluded) from
# the 356 published-and-never-questioned groups of the alpha.20260821a live
# re-baseline. Same mechanism: enumerate_audit.py --all -> fetch_frames.sh ->
# sheet_all.py contact sheets.
#
# Verdicts: 28 SET / 17 PROMO / 5 FURN / 3 TMPL. What that means:
#   * ~9% of the kept pool is genuine furniture the strata could not see —
#     all LOW-coverage (an "<NETWORK> DAILY" logo family at coverage ~0.0004,
#     another break-countdown-slate variant, a Webb-telescope break teaser,
#     a second <NETWORK> PSA instance set, and reality_b's own title card,
#     sibling of the verified reality_a card). Extrapolated: ~30 furniture groups
#     among the 356. All are PUBLISHED and all read hash_confirmed: true, so
#     nothing is lost — this measures what the coverage/entropy signatures
#     miss, not a pipeline defect.
#   * PROMO dominates the news-asset small groups: 13 of news_a's 15
#     sampled groups are streamer promo spots re-airing ~6x across the
#     2-hour block. Genuine recurrence, correctly published, correctly
#     hash-confirmed (the same render re-airs).
#   * The hash flag tracks the semantics exactly where it should: all 5
#     FURN and all re-aired PROMOs read true; the 8 false flags in the
#     sample are roaming set cameras (quiz_a wides, cooking_a kitchen,
#     factual_c audience) and one reconstruction-footage variant — different
#     renders of the same place, which is precisely what false means.
AUDIT_SAMPLE_LABELS = [
    ("news_a", "SET",
     {190, 1775, 1873}),
    ("factual_c", "SET",
     {459, 489, 517}),
    ("news_a", "PROMO",
     {991, 1582, 2017}),
    ("reality_c", "SET",
     {293, 380, 410}),
    ("film_b", "SET",
     {2176, 2193, 2212, 2217}),
    ("sports_doc_a", "TMPL",  # Italia-90 group-standings archival graphic, data differs per airing
     {373, 453, 478}),
    ("quiz_a", "SET",  # bank-total set shot; LED money value differs, hence hash_confirmed false
     {165, 312, 430, 556, 667}),
    ("news_a", "PROMO",
     {322, 423, 860, 961, 1552, 1655, 1987}),
    ("sports_doc_a", "SET",
     {90, 568, 1192}),
    ("docu_a", "PROMO",  # Concorde crash-reconstruction b-roll reused across the episode
     {13, 877, 961}),
    ("news_a", "PROMO",
     {401, 939, 1633, 1964}),
    ("sports_doc_a", "SET",
     {329, 331, 561, 819, 834, 837}),
    ("quiz_a", "SET",
     {135, 293, 404}),
    ("quiz_a", "SET",
     {78, 179, 203}),
    ("docu_a", "PROMO",
     {171, 683, 900}),
    ("news_a", "PROMO",  # streamer promo spot re-aired ~6x — one of 13 promo-spot groups in this sample
     {367, 468, 905, 1009, 1599, 1703, 1930, 2035}),
    ("markets_a", "SET",
     {17, 20, 22, 28}),
    ("drama_f", "SET",
     {71, 802, 879}),
    ("markets_a", "TMPL",  # "DATA CHECK" panel, second instance set
     {68, 318, 338}),
    ("news_a", "PROMO",
     {384, 922, 1615, 1947}),
    ("studio_b3", "SET",
     {282, 323, 375, 379}),
    ("quiz_a", "SET",
     {223, 227, 231, 363, 367, 371, 668, 719}),
    ("quiz_a", "SET",
     {57, 74, 341}),
    ("news_b", "FURN",  # "<NETWORK> DAILY" logo cards — a furniture family the strata missed (coverage ~0.0004)
     {432, 467, 1281, 1344}),
    ("news_a", "PROMO",
     {372, 473, 910, 1014, 1604, 1708, 1935, 2040}),
    ("news_b", "FURN",  # network logo plate WITH break countdown — another break-slate variant
     {796, 1109, 1248, 1350}),
    ("news_a", "PROMO",
     {1005, 1699, 2031}),
    ("news_a", "PROMO",
     {1006, 1700, 2032}),
    ("studio_b1", "SET",
     {303, 422, 475}),
    ("news_b", "SET",
     {41, 45, 53, 60, 84}),
    ("markets_a", "TMPL",  # "COMING UP" guest-lineup boards, guests differ per airing
     {193, 310, 411}),
    ("docu_a", "PROMO",
     {118, 761, 880}),
    ("reality_c", "SET",
     {167, 200, 205, 207, 331, 334, 713}),
    ("news_a", "PROMO",
     {327, 428, 865, 966, 1660, 1992}),
    ("news_b", "FURN",  # Webb-telescope teaser footage with countdown chip — break-slate family member
     {130, 220, 336, 905}),
    ("docu_a", "PROMO",
     {146, 680, 887}),
    ("news_a", "FURN",  # <NETWORK> PSA star card, second instance set
     {316, 854, 1546, 1627, 1981}),
    ("quiz_a", "SET",
     {247, 514, 634}),
    ("news_a", "PROMO",
     {360, 461, 898, 1001, 1592, 1695, 1923, 2027}),
    ("docu_a", "SET",
     {36, 738, 914, 920, 934, 977, 999, 1007}),
    ("news_a", "PROMO",
     {364, 465, 902, 1596, 1927}),
    ("sports_doc_a", "SET",
     {452, 546, 1183}),
    ("sports_doc_a", "SET",
     {523, 555, 1170, 1190}),
    ("news_a", "PROMO",
     {362, 463, 900, 1003, 1594, 1697, 1925, 2029}),
    ("news_a", "PROMO",
     {309, 847, 1539}),
    ("factual_c", "SET",
     {8, 908, 910, 1076, 1091}),
    ("quiz_a", "SET",
     {257, 279, 407, 417, 534, 541, 642}),
    ("reality_c", "SET",
     {116, 195, 197, 363}),
    ("reality_b", "FURN",  # TITLE CARD — sibling of the verified reality_a card
     {1, 51, 1023}),
    ("reality_a", "SET",
     {70, 601, 663}),
    ("studio_b2", "SET",
     {56, 61, 574, 601, 605, 791}),
    ("cooking_a", "SET",
     {133, 276, 392}),
    ("arts_doc_a", "SET",
     {172, 291, 327, 330}),
]

# Fold the sample into the same accessor: label_audited() checks both tables.
AUDIT_LABELS.extend(AUDIT_SAMPLE_LABELS)
