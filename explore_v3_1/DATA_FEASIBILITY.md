# Independent-data feasibility: improved reconstruction, confirmation still gated

11 September 2026. No acquired-panel model predictions or outcome-based model
selection were performed. See `data_feasibility.json`, corrected by
`alphabet_audit.json` for RNA/DNA-normalized homology and final eligibility.

## OPED: the linkage gap is now more specific

Acquired the sequencing-primer supplement, the repository file-tree metadata,
and XML descriptions for all 30 sequencing experiments. The supplement contains
30 gene-level primer pairs, not a sample barcode map. Experiment descriptions
identify biological replicate numbers and lists of pooled loci/editor conditions;
some also identify pooled design tools. They do not resolve which read barcodes
belong to each individual design and editing condition. Conditions cannot be
inferred from editing frequency or sample ordering.

For example, SRX17662522 describes replicate 1 across 30 loci with PE2/ePE and
untreated material; SRX20771652 describes six loci with PE3/ePE. Such a run is not
a single labelled design. The public repository tree contains no clearly labelled
barcode/sample-to-design assignment file. This is a bounded search, not proof
that a mapping does not exist anywhere.

**Decision:** no approximately 49.4-GB raw-read download. The required next input
is the demultiplexing/condition/design map, or already linked quantitative
outcomes with provenance. No author was contacted. Sources: [OPED data and
supplements](https://www.nature.com/articles/s42256-023-00739-w),
[representative ENA experiment](https://www.ebi.ac.uk/ena/browser/view/SRX17662522),
[authors' repository](https://github.com/wenjiegroup/OPED).

## ePRIDICT: 140 input candidates, still only 15 decisions

The previous audit resolved 143 of 146 designs to an unambiguous intended allele.
Joint spacer/nick/PBS/RTT reconstruction now succeeds uniquely for 142 of those
143. Cropped WT/edited windows preserve the full intended change and fit the
model's 100 aligned-token and 90 pegRNA-token limits. Full amplicons must not be
passed through head/tail truncation, which can remove their central edit.

The initial new spacer-homology pass was incomplete because it filtered out RNA
`U` strings. The corrected pass normalizes RNA/DNA before comparison and covers
38,814 unique source spacer cores. There are no exact core-19 matches. Combining
the prespecified <=2-mismatch core/flank checks flags two reconstructed designs.
These checks are local sequence-homology screens, not whole-genome homology.

| Corrected, token-safe population | Designs | Intended alleles | Multi-design groups | Maximum depth |
|---|---:|---:|---:|---:|
| K562 | 140 | 111 | 15 | 3 |
| HEK293T | 54 | 54 | 0 | 1 |

The arrayed experiment methods specify PE2. Input orientation, crop and legacy
RNA encoding must remain explicit; scaffold/context mapping and end-to-end
deployment validation still precede any outcome evaluation. The panel is not
two independent studies, and its 15 shallow choices cannot precisely establish
a small universal gain. [Arrayed methods and supplementary outcomes](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617539/).

For a transparent precision illustration, even treating all 15 decisions as
independent, a conventional paired-t interval with half-width 0.001 would
require a paired-difference standard deviation below approximately 0.00181
(`0.001 * sqrt(15) / t_0.975,14`). This is an optimistic planning calculation,
not measured precision or a power estimate; actual paired differences have
not been scored, and locus dependence would reduce effective sample size.
It does not imply that 15 decisions can never detect a large consistent effect.

## A checkpoint-compatibility issue, not a changed historical result

All source spacer/PBS/RTT fields use A/C/G/U with no true N or T. The historical
tokenizer assigns U the token named N. This still distinguishes all four observed
bases, but sending DNA T to the same checkpoint uses a different token. V3's
source and target caches both use the RNA convention and contain zero pegRNA T
tokens. Thus this discovery does not invalidate their matched comparisons or
justify replacing their recorded results. It does require an explicit legacy
input adapter for new DNA-formatted panels, or consistently retrained models
under a new tokenizer version—not an inference-only vocabulary swap.

The original allele/full-spacer component exclusion also passes source-audit
isolation after normalization. A stronger core-19 check finds one target outer
validation component (one group, eight candidates) matching a source replay
core, despite no full-spacer or allele match. Preserve the frozen primary
population and report an eight-candidate component-excluded sensitivity.
`core_overlap_audit.json` identifies the component and affected source records.
Completed screen sensitivity analysis excludes the whole affected component
(5,560 rather than 5,561 target groups) and leaves the qualitative comparisons
unchanged. None of the matching source rows is in the eligible informative
replay pool (`secondary_results.json`). The broader source-pretraining exposure
still warrants reporting the sensitivity.

## Controlled-transfer and release feasibility

The source inventory covers 20 study/cell/editor contexts. Counts and decision
depth are in `data_feasibility.json`; these are candidate holdouts, not unseen
data for the existing checkpoint. A genuine leave-domain-out comparison requires
new pretraining excluding the domain and overlapping/homologous components.

The official source corpus was supplied privately. No records or sequence-level
derivatives were uploaded or redistributed. Public release still requires
permission or an auditable public-data reconstruction.

**Independent confirmation remains unavailable.** Complete the bounded model
study and use its stop/advance gate; do not represent a re-split Kim validation,
the small endogenous panel, or an exposed source audit as fresh confirmation.
