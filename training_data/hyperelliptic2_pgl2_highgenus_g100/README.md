# hyperelliptic2 PGL2 High-Genus Seed Data

Small tracked seed dataset for `hyperelliptic2` runs.

This is a rebalanced subset of the PGL2 orbit seed data:

- low genus capped per prime
- genus `>= 5` kept from the source PGL2 seed set
- intended for fixed-prime runs, especially `--p 3`

Use with:

```bash
--initial_data_sqlite training_data/hyperelliptic2_pgl2_highgenus_g100/train_data.pkl
```
