# Starter project validation

The delivered code passed 12 focused unittest checks and one complete command-line
smoke run in the development environment before delivery. Checks cover known URL
feature values, entropy, IP/port parsing, canonicalization, malformed input,
offline extraction, cleaning and conflicting labels, repeatable domain-disjoint
splits, known confusion-matrix metrics, training-only scaling, and result export.
The command-line check also verified a completed status and the evidence archive.

Validation used temporary synthetic fixtures, not the real URL-Phish dataset.
Temporary model files, plots, and fixture data were removed after the checks.
No validation-fixture metrics are included as research findings in this bundle.

The development environment was Linux x86_64 with Python 3.12. Mac execution and
actual source dataset ingestion remain to be verified on the user's MacBook.
Run the same tests locally before starting the dataset experiment. Record the
actual run outputs for the report; do not describe these development checks as
completed experiments on public phishing data.
