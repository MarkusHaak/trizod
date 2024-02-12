# TriZOD

## Description

## Installation

## Usage

## Datasets

The folder ./datasets/latest/ contains ready-to-use datasets that where created by running TriZOD on the full BMRB database using four different filtering presets, namely `unfiltered`, `tolerant`, `moderate` and `strict`. Since the stringency of the filtering criteria are monotonically increasing from unfiltered to strict, the respective dataset sizes are decreasing and each dataset is a superset of those with more stringent criteria. However, since filtering also affects which and how the data is processed, the computed scores can differ for the same entry in two differently filtered TriZOD datasets. Therefore, the data should always be loaded from the respective dataset file. These are given in two formats: The .csv files contain less information but are potentially simpler to read than the .json files. However, the latter contain additional information such as sample conditions.

Sequence redundancy reduction was performed using mmseqs2 on the filtered datasets. The test set was selected from the strict dataset only assuring assuring a maximum of 50% sequence identity with 80% coverage inbetween test set members. The IDs and peptide sequences of the test set members are found in the fasta file ./datasets/latest/TriZOD\_test\_set.fasta. All entries with less than 30% sequence identity to test set entries at 80% coverage were clustered with a 50% sequence identity threshold at 80% coverage in an iterative cluster-update approach, starting with the strict dataset members. The resulting redundancy reduced cluster representatives for the `unfiltered`, `tolerant`, `moderate` and `strict` datasets can be found in respective \<filter-default\>\_rest\_set.fasta files.

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
| ionic-strength-range | [-inf,+inf] | [0,5] | [0,3] | [0,3] |
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
| chemical-denaturants | [] | ['guanidin', 'GdmCl', 'Gdn-Hcl','urea','BME','2-ME','mercaptoethanol'] | ['guanidin', 'GdmCl', 'Gdn-Hcl','urea','BME','2-ME','mercaptoethanol'] | ['guanidin', 'GdmCl', 'Gdn-Hcl','urea','BME','2-ME','mercaptoethanol', 'TFA', 'trifluoroethanol', 'Potassium Pyrophosphate', 'acetic acid', 'CD3COOH', 'DTT', 'dithiothreitol', 'dss', 'deuterated sodium acetate'] |
| exp-method-whitelist | ['', '.'] | ['','solution', 'structures'] | ['','solution', 'structures'] | ['solution', 'structures'] |
| exp-method-blacklist | [] | ['solid', 'state'] | ['solid', 'state'] | ['solid', 'state'] |
| max-offset | +inf | 3 | 3 | 2 |
| reject-shift-type-only | Yes | Yes | No | No |

Please note that each of these filters can be set individually with respective command line options and that this will take precedence over the filter defaults set by the `--filter-defaults` option.

## Support

## Contributing

## Authors and acknowledgment

## License

## Project status
Under active development
