# gate/truth — LOCAL-ONLY

Truth lists are curated people lists for the local measurement corpus.
They name real productions and real cast/talent, so they are never
committed (repo rule: no customer/title identifiers in content). The
scorer reads them from THIS directory (gitignored — untracked but
durable; override with `PB_TRUTH_DIR`), keyed by the local corpus directory name
(`/tmp/gt/in_{asset}`). Shape:

    {"note": "...", "people": [{"name": "...", "aliases": ["..."]}]}

`describer_cues.json` in the same directory is the professional-describer
vocabulary source (character-name oracle); `describer_assets.json` maps
it to its local assets.
