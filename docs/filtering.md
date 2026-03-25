# TriZOD Filtering Reference

TriZOD filters peptide shift data entries from the BMRB database using
configurable criteria. The `--filter-defaults` CLI argument sets default values
for all filters at one of four stringency levels: `unfiltered`, `tolerant`,
`moderate`, and `strict`. Individual filters can be overridden with their
respective CLI arguments.

## Filter Descriptions

| Filter                       | Description                                                                                                                                |
| :--------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| temperature-range            | Minimum and maximum temperature in Kelvin.                                                                                                 |
| ionic-strength-range         | Minimum and maximum ionic strength in Mol.                                                                                                 |
| pH-range                     | Minimum and maximum pH.                                                                                                                    |
| unit-assumptions             | Assume units for Temp., Ionic str. and pH if they are not given and exclude entries instead.                                               |
| unit-corrections             | Correct values for Temp., Ionic str. and pH if units are most likely wrong.                                                                |
| default-conditions           | Assume standard conditions if pH (7), ionic strength (0.1 M) or temperature (298 K) are missing and exclude entries instead.               |
| peptide-length-range         | Minimum (and optionally maximum) peptide sequence length.                                                                                  |
| min-backbone-shift-types     | Minimum number of different backbone shift types (max 7).                                                                                  |
| min-backbone-shift-positions | Minimum number of positions with at least one backbone shift.                                                                              |
| min-backbone-shift-fraction  | Minimum fraction of positions with at least one backbone shift.                                                                            |
| max-noncanonical-fraction    | Maximum fraction of non-canonical amino acids (X count as arbitrary canonical) in the amino acid sequence.                                 |
| max-x-fraction               | Maximum fraction of X letters (arbitrary canonical amino acid) in the amino acid sequence.                                                 |
| keywords-blacklist           | Exclude entries with any of these keywords mentioned anywhere in the BMRB file, case ignored.                                              |
| chemical-denaturants         | Exclude entries with any of these chemicals as substrings of sample components, case ignored.                                              |
| exp-method-whitelist         | Include only entries with any of these keywords as substring of the experiment subtype, case ignored.                                      |
| exp-method-blacklist         | Exclude entries with any of these keywords as substring of the experiment subtype, case ignored.                                           |
| max-offset                   | Maximum valid offset correction for any random coil chemical shift type.                                                                   |
| reject-shift-type-only       | Upon exceeding the maximal offset set by `--max-offset`, exclude only the backbone shifts exceeding the offset instead of the whole entry. |

## Filter Defaults by Stringency Level

| Filter                       | unfiltered  | tolerant                                | moderate                                | strict                                                                                                                                                                                                              |
| :--------------------------- | ----------- | --------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| temperature-range            | [-inf,+inf] | [263,333]                               | [273,313]                               | [273,313]                                                                                                                                                                                                           |
| ionic-strength-range         | [0,+inf]    | [0,7]                                   | [0,5]                                   | [0,3]                                                                                                                                                                                                               |
| pH-range                     | [-inf,+inf] | [2,12]                                  | [4,10]                                  | [6,8]                                                                                                                                                                                                               |
| unit-assumptions             | Yes         | Yes                                     | Yes                                     | No                                                                                                                                                                                                                  |
| unit-corrections             | Yes         | Yes                                     | No                                      | No                                                                                                                                                                                                                  |
| default-conditions           | Yes         | Yes                                     | Yes                                     | No                                                                                                                                                                                                                  |
| peptide-length-range         | [5,+inf]    | [5,+inf]                                | [10,+inf]                               | [15,+inf]                                                                                                                                                                                                           |
| min-backbone-shift-types     | 1           | 2                                       | 3                                       | 5                                                                                                                                                                                                                   |
| min-backbone-shift-positions | 3           | 3                                       | 8                                       | 12                                                                                                                                                                                                                  |
| min-backbone-shift-fraction  | 0.0         | 0.0                                     | 0.6                                     | 0.8                                                                                                                                                                                                                 |
| max-noncanonical-fraction    | 1.0         | 0.1                                     | 0.025                                   | 0.0                                                                                                                                                                                                                 |
| max-x-fraction               | 1.0         | 0.2                                     | 0.05                                    | 0.0                                                                                                                                                                                                                 |
| keywords-blacklist           | []          | ['denatur']                             | ['denatur', 'unfold', 'misfold']        | ['denatur', 'unfold', 'misfold', 'interacti', 'bound']                                                                                                                                                              |
| chemical-denaturants         | []          | ['guanidin', 'GdmCl', 'Gdn-Hcl','urea'] | ['guanidin', 'GdmCl', 'Gdn-Hcl','urea'] | ['guanidin', 'GdmCl', 'Gdn-Hcl', 'urea', 'TFA', 'trifluoroethanol', 'Potassium Pyrophosphate'] |
| exp-method-whitelist         | ['', '.']   | ['','solution', 'structures']           | ['','solution', 'structures']           | ['solution', 'structures']                                                                                                                                                                                          |
| exp-method-blacklist         | []          | ['solid']                               | ['solid']                               | ['solid']                                                                                                                                                                                                           |
| max-offset                   | +inf        | 3                                       | 3                                       | 2                                                                                                                                                                                                                   |
| reject-shift-type-only       | Yes         | Yes                                     | No                                      | No                                                                                                                                                                                                                  |

Each filter can be set individually with the respective CLI option, which takes
precedence over `--filter-defaults`.
