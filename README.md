# TriZOD

Novel, continuous, per-residue disorder scores from protein NMR experiments stored in the BMRB database.

## Description

Accurate quantification of intrinsic disorder, crucial for understanding functional protein dynamics, remains challenging. We introduce TriZOD, an innovative scoring system for protein disorder analysis, utilizing nuclear magnetic resonance (NMR) spectroscopy chemical shifts. Traditional methods provide binary, residue-specific annotations, missing the complex spectrum of protein disorder. TriZOD extends the CheZOD scoring framework with quantitative statistical descriptors, offering a nuanced analysis of intrinsically disordered regions. It calculates per-residue scores from chemical shift data of polypeptides in the Biological Magnetic Resonance Data Bank (BMRB). The CheZOD Z-score is a quantitative metric for how much a set of experimentally determined chemical shifts deviate from random coil chemical shifts. The TriZOD G-scores extend upon them to be independent of the number of available chemical shifts. They are normalized to range between 0 and 1, which is beneficial for interpretation and use in training disorder predictors. Additionally, TriZOD introduces a refined, automated selection of BMRB datasets, including filters for physicochemical properties, keywords, and chemical denaturants. We calculated G-scores for over 15,000 peptides in the BMRB, approximately 10-fold the size of previously published CheZOD datasets.
Validation against DisProt annotations demonstrates substantial agreement yet highlights discrepancies, suggesting the need to reevaluate some disorder annotations. TriZOD advances protein disorder prediction by leveraging the full potential of the BMRB database, refining our understanding of disorder, and challenging existing annotations.

## Installation

## Usage

## Datasets

The latest dataset is published under the DOI [10.6084/m9.figshare.25792035](https://www.doi.org/10.6084/m9.figshare.25792035).

This publication consists of four nested datasets of increasing filter stringency: Unfiltered, tolerant, moderate and strict. An overview of the applied filters is given below. The .json files contain all entries of the BMRB that are in accordance with the given filter levels. These are not redundancy reduced and also contain the test set entries and are therefore not intended for direct use as training sets in machine learning applications. Instead, for this purpose, please use only those entries with IDs found in the [filter_level]_rest_set.fasta files and extract the corresponding information such as TriZOD G-scores and/or physicochemical properties from the respective .json files. These fasta files contain the cluster representatives of the redundancy reduction procedure which was performed in an iterative fashion such that clusters with members found in all filter levels are shared among them and have the same cluster representatives. If necessary, all other cluster members can be retrieved from the given [filter_level]_rest_clu.tsv files. The file TriZOD_test_set.fasta contains the IDs and sequences of the TriZOD test set. It is intended that the corresponding data is taken from the strict dataset.

### Filter defaults

TriZOD filters the peptide shift data entries in the BMRB database given a set of filter criteria. Though these criteria can be set individually with corresponding command line arguments, it is most convinient to use one of four filter default options to adapt the overall stringency of the filters. The command line argument `--filter-defaults` sets default values for all data filtering criteria. The accepted options with increasing stringency are `unfiltered`, `tolerant`, `moderate` and `strict`. The affected filters are:

| Filter | Description | 
| :--- | --- |
| temperature-range | Minimum and maximum temperature in Kelvin. |
| ionic-strength-range | Minimum and maximum ionic strength in Mol. |
| pH-range | Minimum and maximum pH. |
| unit-assumptions | Assume units for Temp., Ionic str. and pH if they are not given and exclude entries instead. |
| unit-corrections | Correct values for Temp., Ionic str. and pH if units are most likely wrong. |
| default-conditions | Assume standard conditions if pH (7), ionic strength (0.1 M) or temperature (298 K) are missing and exclude entries instead. |
| peptide-length-range | Minimum (and optionally maximum) peptide sequence length. |
| min-backbone-shift-types | Minimum number of different backbone shift types (max 7). |
| min-backbone-shift-positions | Minimum number of positions with at least one backbone shift. |
| min-backbone-shift-fraction | Minimum fraction of positions with at least one backbone shift. |
| max-noncanonical-fraction | Maximum fraction of non-canonical amino acids (X count as arbitrary canonical) in the amino acid sequence. |
| max-x-fraction | Maximum fraction of X letters (arbitrary canonical amino acid) in the amino acid sequence. |
| keywords-blacklist | Exclude entries with any of these keywords mentioned anywhere in the BMRB file, case ignored. |
| chemical-denaturants | Exclude entries with any of these chemicals as substrings of sample components, case ignored. |
| exp-method-whitelist | Include only entries with any of these keywords as substring of the experiment subtype, case ignored. |
| exp-method-blacklist | Exclude entries with any of these keywords as substring of the experiment subtype, case ignored. |
| max-offset | Maximum valid offset correction for any random coil chemical shift type. |
| reject-shift-type-only | Upon exceeding the maximal offset set by <--max-offset>, exclude only the backbone shifts exceeding the offset instead of the whole entry. |

The following table lists the respective filtering criteria for each of the four filter default options:

| Filter | unfiltered | tolerant | moderate | strict |
| :--- | --- | --- | --- | --- |
| temperature-range | [-inf,+inf] | [263,333] | [273,313] | [273,313] |
| ionic-strength-range | [0,+inf] | [0,7] | [0,5] | [0,3] |
| pH-range | [-inf,+inf] | [2,12] | [4,10] | [6,8] |
| unit-assumptions | Yes | Yes | Yes | No |
| unit-corrections | Yes | Yes | No | No |
| default-conditions | Yes | Yes | Yes | No |
| peptide-length-range | [5,+inf] | [5,+inf] | [10,+inf] | [15,+inf] |
| min-backbone-shift-types | 1 | 2 | 3 | 5 |
| min-backbone-shift-positions | 3 | 3 | 8 | 12 |
| min-backbone-shift-fraction | 0.0 | 0.0 | 0.6 | 0.8 |
| max-noncanonical-fraction | 1.0 | 0.1 | 0.025 | 0.0 |
| max-x-fraction | 1.0 | 0.2 | 0.05 | 0.0 |
| keywords-blacklist | [] | ['denatur'] | ['denatur', 'unfold', 'misfold'] | ['denatur', 'unfold', 'misfold', 'interacti', 'bound'] |
| chemical-denaturants | [] | ['guanidin', 'GdmCl', 'Gdn-Hcl','urea'] | ['guanidin', 'GdmCl', 'Gdn-Hcl','urea'] | ['guanidin', 'GdmCl', 'Gdn-Hcl','urea','BME','2-ME','mercaptoethanol', 'TFA', 'trifluoroethanol', 'Potassium Pyrophosphate', 'acetic acid', 'CD3COOH', 'DTT', 'dithiothreitol', 'dss', 'deuterated sodium acetate'] |
| exp-method-whitelist | ['', '.'] | ['','solution', 'structures'] | ['','solution', 'structures'] | ['solution', 'structures'] |
| exp-method-blacklist | [] | ['solid', 'state'] | ['solid', 'state'] | ['solid', 'state'] |
| max-offset | +inf | 3 | 3 | 2 |
| reject-shift-type-only | Yes | Yes | No | No |

Please note that each of these filters can be set individually with respective command line options and that this will take precedence over the filter defaults set by the `--filter-defaults` option.

## Project status
Under active development
