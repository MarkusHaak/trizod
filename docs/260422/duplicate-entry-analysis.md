# Duplicate Entry Analysis: Same Protein, Different Experiments

**Tier:** tolerant | **Condition matching:** T within 10K, pH within 1.0

- **5393** entries across **1939** sequences with 2+ entries

## Most frequently measured proteins

| Protein                     | Len | Entries | Single | Bound |
| --------------------------- | --: | ------: | -----: | ----: |
| alpha-synuclein             | 140 |      62 |     57 |     1 |
| entity_1                    | 148 |      46 |      0 |    30 |
| entity_1                    | 133 |      45 |      0 |    14 |
| entity_2                    | 129 |      42 |      0 |    12 |
| human TSG-6                 |  98 |      33 |     14 |    18 |
| entity_1                    |  76 |      31 |     19 |     1 |
| Protein-export protein SecB | 155 |      28 |      0 |     0 |
| AbpSH3                      |  59 |      25 |      1 |     0 |
| entity_1                    | 315 |      23 |      0 |    23 |
| entity_1                    |  21 |      20 |      1 |     0 |

## Bound vs Unbound Shift Differences

### Bound vs Unbound

**581 pairs compared**

| Atom | Pairs | Mean abs | Median abs | Max abs |   Std | >0.5 ppm | >1.0 ppm |
| ---- | ----: | -------: | ---------: | ------: | ----: | -------: | -------: |
| C    |   246 |    0.399 |      0.231 |   4.039 | 0.675 |    19.1% |    10.2% |
| CA   |   398 |    0.481 |      0.330 |   4.235 | 0.655 |    22.5% |    12.3% |
| CB   |   384 |    0.532 |      0.349 |   5.081 | 0.782 |    24.5% |    12.2% |
| HA   |   299 |    0.096 |      0.057 |   0.930 | 0.162 |     3.6% |     0.8% |
| H    |   542 |    0.139 |      0.079 |   1.358 | 0.245 |     5.9% |     1.7% |
| N    |   492 |    0.763 |      0.453 |   7.926 | 1.327 |    32.8% |    19.0% |
| HB   |   295 |    0.088 |      0.048 |   0.768 | 0.149 |     2.7% |     0.6% |

## Independent Measurement Reproducibility

### Independent measurements (noise floor)

**1301 pairs compared**

| Atom | Pairs | Mean abs | Median abs | Max abs |   Std | >0.5 ppm | >1.0 ppm |
| ---- | ----: | -------: | ---------: | ------: | ----: | -------: | -------: |
| C    |   454 |    0.547 |      0.472 |   2.606 | 0.366 |     9.0% |     5.8% |
| CA   |   784 |    0.298 |      0.206 |   2.757 | 0.452 |    13.2% |     8.2% |
| CB   |   736 |    0.305 |      0.205 |   2.635 | 0.453 |    13.8% |     7.4% |
| HA   |   834 |    0.045 |      0.030 |   0.258 | 0.063 |     1.4% |     0.3% |
| H    |  1219 |    0.069 |      0.049 |   0.366 | 0.090 |     3.0% |     0.8% |
| N    |   880 |    0.356 |      0.229 |   3.119 | 0.572 |    15.8% |     8.9% |
| HB   |   812 |    0.101 |      0.056 |   0.442 | 0.137 |     1.5% |     0.5% |

## Binding effect vs experimental noise

| Atom | Binding (MAD) | Noise (MAD) | Ratio |
| ---- | ------------: | ----------: | ----: |
| C    |         0.399 |       0.547 |  0.7x |
| CA   |         0.481 |       0.298 |  1.6x |
| CB   |         0.532 |       0.305 |  1.7x |
| HA   |         0.096 |       0.045 |  2.2x |
| H    |         0.139 |       0.069 |  2.0x |
| N    |         0.763 |       0.356 |  2.1x |
| HB   |         0.088 |       0.101 |  0.9x |

Ratio > 1 means binding effect exceeds experimental noise.

## Example bound/unbound pairs (first 15)

| Protein                   |   Single |    Bound | T_s | T_b | pH_s | pH_b | CA MAD | N MAD |
| ------------------------- | -------: | -------: | --: | --: | ---: | ---: | -----: | ----: |
| derCD23                   |  bmr6734 |  bmr6733 | 308 | 308 |  6.8 |  6.8 |      — | 0.151 |
| derCD23                   |  bmr6735 |  bmr6733 | 308 | 308 |  6.8 |  6.8 |      — | 0.186 |
| MLU1-BOX binding protein  |  bmr4254 |  bmr4256 | 288 | 288 |  7.6 |  7.6 |  0.506 | 0.460 |
| Nitrogen regulatory prote |  bmr4527 |  bmr4528 | 298 | 298 |  6.8 |  6.8 |  0.340 | 0.824 |
| hL-FABP                   | bmr19189 | bmr19188 | 298 | 298 |  6.5 |  6.5 |  0.318 | 1.519 |
| hL-FABP                   | bmr19189 | bmr19160 | 298 | 298 |  6.5 |  6.5 |  0.331 | 1.342 |
| hL-FABP                   | bmr25333 | bmr19188 | 298 | 298 |  6.5 |  6.5 |  2.725 | 1.489 |
| hL-FABP                   | bmr25333 | bmr19160 | 298 | 298 |  6.5 |  6.5 |  2.541 | 1.300 |
| entity_1                  | bmr50387 | bmr50388 | 298 | 298 |  6.5 |  6.5 |  0.141 | 0.361 |
| CAMP_RECEPTOR_PROTEIN     | bmr19144 | bmr19145 | 305 | 305 |  6.0 |  6.0 |  0.147 | 0.236 |
| FKBP12                    | bmr27738 | bmr27739 | 273 | 273 |  6.0 |  6.0 |  0.180 | 0.391 |
| FKBP12                    | bmr16925 | bmr16931 | 298 | 298 |  7.0 |  7.0 |  0.435 | 0.903 |
| FKBP12                    | bmr16925 | bmr16933 | 298 | 298 |  7.0 |  7.0 |  0.533 | 0.689 |
| FKBP12                    | bmr19241 | bmr16931 | 298 | 298 |  6.5 |  7.0 |  0.316 | 1.036 |
| FKBP12                    | bmr19241 | bmr16933 | 298 | 298 |  6.5 |  7.0 |  0.324 | 0.762 |
