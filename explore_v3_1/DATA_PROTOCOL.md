# Outcome-independent follow-up data checks

11 September 2026; specified before the new sequence checks. No external-panel
model scoring or model selection is authorized by this audit script.

OPED: parse the 30 experiment XML descriptions and sequencing primer supplement.
Accept only an explicit design/condition/replicate-to-read linkage, not an
inference from a generic sample label, ordering, or observed editing frequency.
Record pooled conditions, unresolved barcodes and metadata contradictions.

ePRIDICT: keep the previous 143 unambiguous allele mappings fixed. Find the guide's
last 19 nt in both orientations of the authors' WT amplicon, with nick 16 nt after
the core's start. Enumerate all nonempty PBS/RTT splits of the extension; retain
only splits whose reverse-complement PBS matches WT immediately before the nick
and whose reverse-complement RTT matches the intended edited sequence immediately
after it. Require the intended edit to be downstream of the nick and included
in that RTT. Reject nonunique reconstructions rather than selecting by outcomes.
Retain reported spacer sequence, including any leading transcriptional G.

For an input-length diagnostic, crop to encompass spacer+PAM, PBS and the entire
RTT-encoded intended edit, with 10 nt added to either flank. Convert the right
endpoint correctly between WT and edited coordinates across the single minimal
edit; reject windows exceeding 100 aligned edit tokens or pegRNAs exceeding 90
tokens including special tokens. Do not feed entire amplicons into the tokenizer:
its head/tail truncation can remove the central intended edit. This crop is a
candidate preprocessing convention, not yet a validated external deployment.

Additional conservative source-overlap sensitivity checks use minimum Hamming
distance <=2 for equal-length spacer cores (19 nt) and concatenated canonical
allele flanks (24 nt), testing both DNA orientations. No outcome-dependent
threshold choice. These test local homology, not exhaustive genome-wide homology.
Record unresolved contexts explicitly. The paper's arrayed methods identify PE2;
do not substitute PEmax or reinterpret a singleton locus as a selection group.

No claim of input readiness until reconstruction, orientation, token preservation
and context mapping are all checked. No raw-read download without verified linking.
