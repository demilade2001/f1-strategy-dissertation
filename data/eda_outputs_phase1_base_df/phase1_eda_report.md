# Phase 1 Exhaustive EDA Report

Exact base_df.csv path used: /Users/demi/Downloads/Dissertation Code Project/Code Base/data/processed/base_df.csv

Loaded shape: 74536 x 56

Mandatory casts applied before any filtering/grouping: is_strategic_stop -> BooleanDtype, sc_active/vsc_active/caution_active -> bool.

## 1. Descriptive statistics for engineered feature set

Note: undercut/overcut threat index and SC/VSC circuit exposure rate each map to two concrete columns and are reported separately.

```
                 feature      mean      std        min        max  missing_pct
0        circuit_sc_rate  0.114340 0.054988   0.025253   0.210526     0.000000
1       circuit_vsc_rate  0.025808 0.019931   0.000000   0.066976     0.000000
2     deg_rate_corrected  0.023348 0.053612  -1.157700   0.769737    10.055544
3   overcut_threat_index  6.220230 7.842950   0.000000  20.000000     0.000000
4      pace_differential  0.496219 4.272360 -59.557000 119.687000     1.597886
5       pit_window_delta 21.221150 3.098847  -4.116129  27.984132    54.829881
6      position_at_stake  0.498543 0.787575   0.000000   7.000000    30.960878
7  undercut_threat_index  7.486407 7.983825   0.000000  20.000000     0.000000
```

## 2. Pit stop lap-number distribution split by is_strategic_stop

Distribution table: is_strategic_stop=True

```
    LapNumber  count       pct          strategy_split
0           2     37 17.209302  is_strategic_stop=True
1           3      3  1.395349  is_strategic_stop=True
2           4      1  0.465116  is_strategic_stop=True
3           7      2  0.930233  is_strategic_stop=True
4           8      4  1.860465  is_strategic_stop=True
5          12      1  0.465116  is_strategic_stop=True
6          14      3  1.395349  is_strategic_stop=True
7          15      3  1.395349  is_strategic_stop=True
8          16      1  0.465116  is_strategic_stop=True
9          18      5  2.325581  is_strategic_stop=True
10         19      2  0.930233  is_strategic_stop=True
11         22      1  0.465116  is_strategic_stop=True
12         23      9  4.186047  is_strategic_stop=True
13         24      1  0.465116  is_strategic_stop=True
14         25     13  6.046512  is_strategic_stop=True
15         26     10  4.651163  is_strategic_stop=True
16         27      6  2.790698  is_strategic_stop=True
17         28      3  1.395349  is_strategic_stop=True
18         29      2  0.930233  is_strategic_stop=True
19         30      7  3.255814  is_strategic_stop=True
20         31      1  0.465116  is_strategic_stop=True
21         32     12  5.581395  is_strategic_stop=True
22         33      5  2.325581  is_strategic_stop=True
23         34     10  4.651163  is_strategic_stop=True
24         35      3  1.395349  is_strategic_stop=True
25         36      8  3.720930  is_strategic_stop=True
26         37     14  6.511628  is_strategic_stop=True
27         38      1  0.465116  is_strategic_stop=True
28         39      2  0.930233  is_strategic_stop=True
29         40      2  0.930233  is_strategic_stop=True
30         45      1  0.465116  is_strategic_stop=True
31         47      2  0.930233  is_strategic_stop=True
32         48      2  0.930233  is_strategic_stop=True
33         49      3  1.395349  is_strategic_stop=True
34         52      1  0.465116  is_strategic_stop=True
35         53      1  0.465116  is_strategic_stop=True
36         54      2  0.930233  is_strategic_stop=True
37         55      1  0.465116  is_strategic_stop=True
38         56      4  1.860465  is_strategic_stop=True
39         57      3  1.395349  is_strategic_stop=True
40         63     11  5.116279  is_strategic_stop=True
41         64     12  5.581395  is_strategic_stop=True
```

Distribution table: is_strategic_stop=False

```
    LapNumber  count      pct           strategy_split
0           2      1 0.258398  is_strategic_stop=False
1           3      1 0.258398  is_strategic_stop=False
2           5      3 0.775194  is_strategic_stop=False
3           7      1 0.258398  is_strategic_stop=False
4           8      2 0.516796  is_strategic_stop=False
5           9      5 1.291990  is_strategic_stop=False
6          10      4 1.033592  is_strategic_stop=False
7          11     11 2.842377  is_strategic_stop=False
8          12      7 1.808786  is_strategic_stop=False
9          13      7 1.808786  is_strategic_stop=False
10         14     11 2.842377  is_strategic_stop=False
11         15      7 1.808786  is_strategic_stop=False
12         16      6 1.550388  is_strategic_stop=False
13         17      8 2.067183  is_strategic_stop=False
14         18      6 1.550388  is_strategic_stop=False
15         19      8 2.067183  is_strategic_stop=False
16         20      4 1.033592  is_strategic_stop=False
17         21      6 1.550388  is_strategic_stop=False
18         22     13 3.359173  is_strategic_stop=False
19         23      7 1.808786  is_strategic_stop=False
20         24      6 1.550388  is_strategic_stop=False
21         25     10 2.583979  is_strategic_stop=False
22         26      6 1.550388  is_strategic_stop=False
23         27     11 2.842377  is_strategic_stop=False
24         28     10 2.583979  is_strategic_stop=False
25         29     10 2.583979  is_strategic_stop=False
26         30     10 2.583979  is_strategic_stop=False
27         31     11 2.842377  is_strategic_stop=False
28         32     11 2.842377  is_strategic_stop=False
29         33     14 3.617571  is_strategic_stop=False
30         34     18 4.651163  is_strategic_stop=False
31         35     11 2.842377  is_strategic_stop=False
32         36     16 4.134367  is_strategic_stop=False
33         37      8 2.067183  is_strategic_stop=False
34         38      7 1.808786  is_strategic_stop=False
35         39      8 2.067183  is_strategic_stop=False
36         40      7 1.808786  is_strategic_stop=False
37         41     10 2.583979  is_strategic_stop=False
38         42      7 1.808786  is_strategic_stop=False
39         43     11 2.842377  is_strategic_stop=False
40         44      8 2.067183  is_strategic_stop=False
41         45      7 1.808786  is_strategic_stop=False
42         46      6 1.550388  is_strategic_stop=False
43         47      4 1.033592  is_strategic_stop=False
44         48      2 0.516796  is_strategic_stop=False
45         49      2 0.516796  is_strategic_stop=False
46         50      2 0.516796  is_strategic_stop=False
47         51      5 1.291990  is_strategic_stop=False
48         52      3 0.775194  is_strategic_stop=False
49         53      3 0.775194  is_strategic_stop=False
50         54      2 0.516796  is_strategic_stop=False
51         55      4 1.033592  is_strategic_stop=False
52         56      1 0.258398  is_strategic_stop=False
53         57     12 3.100775  is_strategic_stop=False
54         58      1 0.258398  is_strategic_stop=False
55         64      4 1.033592  is_strategic_stop=False
56         67      1 0.258398  is_strategic_stop=False
```

Figure saved: /Users/demi/Downloads/Dissertation Code Project/Code Base/data/eda_outputs_phase1_base_df/figures/pit_stop_lap_distribution_by_strategy.png

## 3. SC/VSC frequency by circuit (2022-2024)

```
                    EventName  row_count  sc_frequency  vsc_frequency  caution_frequency
8          Chinese Grand Prix       1032      0.180233       0.035853           0.216085
19           Qatar Grand Prix       1949      0.181119       0.000000           0.181119
7         Canadian Grand Prix       3853      0.152868       0.027771           0.180638
23       São Paulo Grand Prix       3503      0.157294       0.017414           0.174707
21       Singapore Grand Prix       3210      0.117757       0.043925           0.161682
11          French Grand Prix        958      0.123173       0.021921           0.145094
1       Australian Grand Prix       3046      0.105384       0.039068           0.144452
15       Las Vegas Grand Prix       1884      0.121019       0.021231           0.142251
18          Monaco Grand Prix       3931      0.116764       0.004325           0.121089
20   Saudi Arabian Grand Prix       2664      0.100225       0.017267           0.117492
3       Azerbaijan Grand Prix       2826      0.068294       0.039986           0.108280
9            Dutch Grand Prix       4161      0.086998       0.019226           0.106224
24   United States Grand Prix       3065      0.100489       0.000000           0.100489
17           Miami Grand Prix       3306      0.083182       0.014217           0.097399
16     Mexico City Grand Prix       3876      0.065789       0.013932           0.079721
14        Japanese Grand Prix       2294      0.067132       0.012206           0.079337
4          Bahrain Grand Prix       3310      0.066163       0.010876           0.077039
6          British Grand Prix       2747      0.069530       0.005825           0.075355
2         Austrian Grand Prix       4083      0.032819       0.026941           0.059760
5          Belgian Grand Prix       2449      0.059616       0.000000           0.059616
12       Hungarian Grand Prix       3921      0.033155       0.023208           0.056363
13         Italian Grand Prix       2937      0.046306       0.007491           0.053796
10  Emilia Romagna Grand Prix       2370      0.047679       0.000000           0.047679
0        Abu Dhabi Grand Prix       3309      0.022363       0.011484           0.033847
22         Spanish Grand Prix       3852      0.015576       0.000000           0.015576
```

Figure saved: /Users/demi/Downloads/Dissertation Code Project/Code Base/data/eda_outputs_phase1_base_df/figures/sc_vsc_frequency_by_circuit.png

## 4. Archetype consistency check on empirical SC/VSC exposure

Reference medians by assigned archetype (empirical caution_frequency):

```
Street       0.114685
Power        0.067486
Technical    0.045105
```

Circuit-level consistency results:

```
   assigned_archetype                circuit  empirical_caution_frequency  own_archetype_median                      consistency_flag
0              Street      Monaco Grand Prix                     0.121089              0.114685                            CONSISTENT
1              Street  Azerbaijan Grand Prix                     0.108280              0.114685                            CONSISTENT
2              Street   Singapore Grand Prix                     0.161682              0.114685                            CONSISTENT
3              Street       Miami Grand Prix                     0.097399              0.114685                            CONSISTENT
4               Power     Italian Grand Prix                     0.053796              0.067486  CONTRADICTION -> closer to Technical
5               Power     Belgian Grand Prix                     0.059616              0.067486                            CONSISTENT
6               Power     British Grand Prix                     0.075355              0.067486                            CONSISTENT
7               Power     Bahrain Grand Prix                     0.077039              0.067486                            CONSISTENT
8           Technical     Spanish Grand Prix                     0.015576              0.045105                            CONSISTENT
9           Technical   Hungarian Grand Prix                     0.056363              0.045105      CONTRADICTION -> closer to Power
10          Technical   Abu Dhabi Grand Prix                     0.033847              0.045105                            CONSISTENT
11          Technical    Japanese Grand Prix                     0.079337              0.045105      CONTRADICTION -> closer to Power
```

### 5. SOFT degradation facets

Figure saved: /Users/demi/Downloads/Dissertation Code Project/Code Base/data/eda_outputs_phase1_base_df/figures/tyre_degradation_curves_soft.png

### 5. MEDIUM degradation facets

Figure saved: /Users/demi/Downloads/Dissertation Code Project/Code Base/data/eda_outputs_phase1_base_df/figures/tyre_degradation_curves_medium.png

### 5. HARD degradation facets

Figure saved: /Users/demi/Downloads/Dissertation Code Project/Code Base/data/eda_outputs_phase1_base_df/figures/tyre_degradation_curves_hard.png

## 6. Missing-data audit across all columns

```
                         column  missing_count  missing_pct
0                       PitTime          74536   100.000000
1                     CarNumber          74536   100.000000
2             is_strategic_stop          73934    99.192337
3                    PitOutTime          71934    96.509069
4                     PitInTime          71910    96.476870
5                   pit_delta_3          40868    54.829881
6                   pit_delta_1          40868    54.829881
7              pit_window_delta          40868    54.829881
8                   pit_delta_5          40868    54.829881
9             position_at_stake          23077    30.960878
10              pressure_weight          22995    30.850864
11                      SpeedI1          11315    15.180584
12                     deg_rate           7504    10.067618
13           deg_rate_corrected           7495    10.055544
14                      SpeedFL           2739     3.674734
15                    Sector1_s           1582     2.122464
16            pace_differential           1191     1.597886
17                    LapTime_s           1191     1.597886
18       fuel_corrected_laptime           1191     1.597886
19                     TyreLife            555     0.744607
20                 TyreCompound            369     0.495063
21                    Sector3_s            233     0.312601
22                      SpeedI2            143     0.191854
23                    Sector2_s            123     0.165021
24                     Position            109     0.146238
25                 points_delta            109     0.146238
26            circuit_archetype              0     0.000000
27              circuit_sc_rate              0     0.000000
28             circuit_vsc_rate              0     0.000000
29                         Year              0     0.000000
30   circuit_sc_events_per_race              0     0.000000
31  circuit_vsc_events_per_race              0     0.000000
32              high_sc_circuit              0     0.000000
33             starting_fuel_kg              0     0.000000
34                 fuel_load_kg              0     0.000000
35         undercut_threat_flag              0     0.000000
36  deg_rate_corrected_reliable              0     0.000000
37          overcut_threat_flag              0     0.000000
38                  is_midfield              0     0.000000
39         overcut_threat_index              0     0.000000
40                   IsAccurate              0     0.000000
41                    EventName              0     0.000000
42                       Driver              0     0.000000
43                    LapNumber              0     0.000000
44                        Stint              0     0.000000
45                    FreshTyre              0     0.000000
46                  TrackStatus              0     0.000000
47                         Team              0     0.000000
48        undercut_threat_index              0     0.000000
49                    sc_active              0     0.000000
50                   vsc_active              0     0.000000
51               compound_known              0     0.000000
52                        Round              0     0.000000
53            deg_rate_reliable              0     0.000000
54                   pit_loss_s              0     0.000000
55               caution_active              0     0.000000
```

Columns flagged above 5% missing:

```
                column  missing_count  missing_pct
0              PitTime          74536   100.000000
1            CarNumber          74536   100.000000
2    is_strategic_stop          73934    99.192337
3           PitOutTime          71934    96.509069
4            PitInTime          71910    96.476870
5          pit_delta_3          40868    54.829881
6          pit_delta_1          40868    54.829881
7     pit_window_delta          40868    54.829881
8          pit_delta_5          40868    54.829881
9    position_at_stake          23077    30.960878
10     pressure_weight          22995    30.850864
11             SpeedI1          11315    15.180584
12            deg_rate           7504    10.067618
13  deg_rate_corrected           7495    10.055544
```

## 7. Outlier audit on core engineered features

Outliers are reported only (no dropping/transformation), using both ±3SD and IQR bounds.

```
                 feature  non_null_n  outlier_3sd_count  outlier_3sd_pct  outlier_iqr_count  outlier_iqr_pct
0        circuit_sc_rate       74536                  0         0.000000                  0         0.000000
1       circuit_vsc_rate       74536                  0         0.000000                  0         0.000000
2     deg_rate_corrected       67041                724         1.079936               3576         5.334049
3   overcut_threat_index       74536                  0         0.000000                  0         0.000000
4      pace_differential       73345               2399         3.270843               7971        10.867816
5       pit_window_delta       33668                532         1.580135               2948         8.756089
6      position_at_stake       51459                617         1.199013               1326         2.576809
7  undercut_threat_index       74536                  0         0.000000                  0         0.000000
```
