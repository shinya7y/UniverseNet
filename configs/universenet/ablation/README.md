# UniverseNet ablation experiments

This directory contains configs for research and analysis.
Please use non-ablated models for practical purposes.

## Results and Models

### Ablation from UniverseNet 20.08

| Method                            | Backbone      | box AP |
| :-------------------------------- | :------------ | :----: |
| UniverseNet 20.08                 | R2-50-v1b     |  47.5  |
| UniverseNet 20.08                 | R2-50 (orig.) |  46.3  |
| UniverseNet 20.08                 | R-50-B        |  44.7  |
| UniverseNet 20.08                 | R-50-C        |  45.8  |
| UniverseNet 20.08 w/o SEPC        | R2-50-v1b     |  45.8  |
| UniverseNet 20.08 w/o DCN         | R2-50-v1b     |  45.9  |
| UniverseNet 20.08 w/o mstrain     | R2-50-v1b     |  45.9  |
| UniverseNet 20.08 w/o SyncBN, iBN | R2-50-v1b     |  45.8  |
