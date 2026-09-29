# Retrieval evaluation

**Question:** which embedding model and which document template give the best semantic
search, measured rather than guessed?

## Method

- **Query set:** [`eval/queries.yaml`](../eval/queries.yaml) has 32 natural-language requests
  that describe premise or mood **without title words** ("a surgeon hunts a charming serial
  killer he once saved"). The score measures semantic understanding, not keyword overlap.
- **Relevance:** each query lists acceptable answers as franchise prefixes (sequels count)
  or exact titles (for ambiguous prefixes such as *Monster* vs *Monster Musume*). A query is
  a hit@k if any relevant anime is in the top k.
- **Metrics:** Hit@1, Hit@5, Hit@10 and MRR@10.
- **Search:** exact brute-force cosine in numpy, so there are no ANN effects and no index
  per configuration.
- **Templates:**
  - `full`: labelled lines with title, format and years, genres, themes, demographic,
    studio and synopsis.
  - `synopsis_only`: plot text only, with the title as a fallback.
- **Corpora:**
  - the **full 9,999 anime**, which is what production searches;
  - the **top 970 by rank**, the subset Gemini covered before the free-tier daily quota
    (1,000 texts) ran out.

Reproduce with:

```bash
make eval GPU=1 ARGS="--models local:BAAI/bge-base-en-v1.5,local:BAAI/bge-large-en-v1.5,local:google/embeddinggemma-300m --templates full,synopsis_only"
```

## Results: full corpus (9,999 anime)

| Model | Template | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| BAAI/bge-base-en-v1.5 | full | 0.47 | 0.66 | 0.78 | 0.553 |
| BAAI/bge-base-en-v1.5 | synopsis_only | 0.56 | 0.88 | 0.91 | 0.697 |
| BAAI/bge-large-en-v1.5 | full | 0.47 | 0.66 | 0.78 | 0.557 |
| BAAI/bge-large-en-v1.5 | synopsis_only | 0.72 | 0.84 | 0.91 | 0.784 |
| google/embeddinggemma-300m | full | 0.69 | 0.91 | 0.97 | 0.776 |
| **google/embeddinggemma-300m** | **synopsis_only** | **0.75** | **0.94** | **1.00** | **0.822** |

## Results: top-970 corpus (includes the Gemini API)

| Model | Template | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| gemini-embedding-001 (API, 768-d) | full | 0.91 | 1.00 | 1.00 | 0.948 |
| BAAI/bge-base-en-v1.5 | full | 0.72 | 0.88 | 0.91 | 0.781 |
| BAAI/bge-base-en-v1.5 | synopsis_only | 0.88 | 0.97 | 1.00 | 0.918 |
| google/embeddinggemma-300m | full | 0.81 | 1.00 | 1.00 | 0.901 |
| **google/embeddinggemma-300m** | **synopsis_only** | **0.91** | **1.00** | **1.00** | **0.953** |

Gemini with `synopsis_only` was not measured: re-embedding the corpus would have taken
another day of free-tier quota.

## Findings

1. **Leaving metadata out of the embedded text helps every model.** `synopsis_only` beats
   `full` for all three open models, by +0.05 to +0.23 MRR. Titles and genre lists pull
   results toward lexical overlap: bge-base answered "similar to Death Note" with *Death
   March…*. Metadata still matters, so it moved to where it works best: Qdrant payload
   filters, and a tag-overlap re-rank for item-to-item similarity.
2. **Model knowledge beats model size.** bge-large (335M) barely beats bge-base (110M) with
   the `full` template. EmbeddingGemma (300M) wins mostly on queries that need world
   knowledge: *spirits-countryside* and *body-swap-comet* are missed by both bge models
   and found by EmbeddingGemma.
3. **Small corpora flatter.** Every model scores lower on 9,999 anime than on the
   best-known 970. The best configuration drops from MRR 0.953 to 0.822. An evaluation on
   the popular head alone would overstate production quality.
4. **The local model matches the hosted API.** EmbeddingGemma (`synopsis_only`, MRR 0.953)
   and the Gemini API (`full`, 0.948) are tied within noise on 32 queries. The difference
   is that Gemini is far more **robust to metadata-heavy text** (0.948 vs 0.901 for
   EmbeddingGemma on `full`), so the template matters much less with it. The local model
   won the deployment decision on cost and operations (no quota, no per-query fee, runs
   offline), not on quality.

## Per-query ranks, full corpus (rank of first relevant result; - = not in top 10)

| Query | bge-base / full | bge-base / syn | bge-large / full | bge-large / syn | gemma / full | gemma / syn |
|---|---|---|---|---|---|---|
| genius-vs-detective | 4 | 3 | 10 | 1 | 7 | 7 |
| rebellion-mastermind | - | 1 | - | - | - | 6 |
| time-travel-save-friends | 8 | 3 | 3 | 1 | 4 | 5 |
| surgeon-serial-killer | 3 | 2 | 1 | 1 | 1 | 1 |
| dystopian-crime-prediction | 1 | 1 | 1 | 2 | 1 | 1 |
| titans-walls | 1 | 1 | 3 | 1 | 1 | 1 |
| boxing-underdog | 1 | 3 | 3 | 1 | 2 | 1 |
| volleyball-short-player | 1 | 1 | 1 | 1 | 1 | 1 |
| camping-girls | 1 | 1 | 1 | 1 | 1 | 1 |
| spirits-countryside | - | - | - | - | 3 | 3 |
| space-bounty-hunters | 1 | 1 | 1 | 2 | 5 | 2 |
| alchemy-brothers | 1 | 1 | 1 | 1 | 1 | 1 |
| vikings-revenge | 1 | 1 | 1 | 1 | 1 | 1 |
| descend-into-abyss | - | 3 | 1 | 1 | 1 | 1 |
| elf-after-hero-party | - | 1 | - | 1 | 1 | 1 |
| deaf-girl-bullying | 1 | 1 | 2 | 1 | 1 | 1 |
| pianist-grief | 2 | 2 | 8 | 1 | 1 | 1 |
| letters-war-veteran | 8 | 3 | - | 8 | 4 | 5 |
| body-swap-comet | - | - | - | - | 2 | 4 |
| bathhouse-spirits | - | 2 | - | 7 | 1 | 1 |
| war-orphans-firebombing | 1 | 2 | 3 | 2 | 1 | 1 |
| spy-family | 1 | 1 | 1 | 1 | 1 | 1 |
| love-war-student-council | 2 | 7 | 10 | 3 | 7 | 1 |
| shy-guitarist | 3 | 1 | 6 | 1 | 1 | 1 |
| magical-girl-dark | - | - | - | 1 | 1 | 2 |
| space-opera-strategy | 1 | 1 | 1 | 1 | 1 | 1 |
| china-warring-states | 1 | 1 | 1 | 1 | 2 | 1 |
| orphans-escape-farm | 1 | 1 | 1 | 1 | 1 | 1 |
| cooking-dungeon | 1 | 1 | 1 | 1 | 1 | 1 |
| apothecary-palace-mysteries | 6 | 1 | 1 | 1 | 1 | 1 |
| antarctica-trip | 10 | 2 | 2 | 1 | 1 | 1 |
| shogi-depression | 4 | 1 | 1 | 1 | 1 | 1 |

## Limitations

- 32 queries is a small set, written by the developer, so there's a risk of bias toward
  phrasings the author finds natural. The per-query table makes each judgement inspectable.
- The set measures **text-to-anime retrieval** only. Item-to-item similarity ("more like
  this") and chat intent parsing aren't measured yet; both are listed as future work.
