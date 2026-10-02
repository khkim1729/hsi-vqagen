# Fixed Feasibility Samples

Every experimental configuration uses the same five sample IDs in the order below. The manifest hash is `e64324facd80724ec65afb0734f09d559d1038d919cf2204f908244851205b9f`; runners must reject a missing, duplicated, or reordered entry.

| Order | Sample ID | Shard | Selection reason |
|---:|---|---|---|
| 1 | `NEON_D01_BART_DP3_312000_4875000_bidirectional_reflectance-468-532X660-724` | `s0000` | Forest canopy with strong vegetation structure. |
| 2 | `NEON_D01_HARV_DP3_725000_4701000_bidirectional_reflectance-404-468X788-852` | `s0054` | Open-water and vegetation boundary. |
| 3 | `NEON_D06_KONZ_DP3_708000_4336000_bidirectional_reflectance-788-852X212-276` | `s0201` | Dry cropland with distinct soil and vegetation signals. |
| 4 | `NEON_D14_SRER_DP3_506000_3520000_bidirectional_reflectance-788-852X532-596` | `s0325` | Shrub and bare surface with a bright linear feature. |
| 5 | `NEON_D19_DEJU_DP3_566000_7088000_bidirectional_reflectance-532-596X724-788` | `s0397` | Bright developed or exposed surface. |

The set was chosen only after checking readable 384 × 384 RGB files, non-empty paired descriptions, 64 × 64 × 426 metadata, and diverse visible/semantic scene characteristics. The selection is not changed per checkpoint or input condition.
