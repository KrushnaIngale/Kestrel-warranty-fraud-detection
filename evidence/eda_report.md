# EDA Report — Kestrel Warranty Fraud Detection

## 1. Dataset Overview

- **Train (raw)**: 12029 rows, 14 columns
- **Train (deduplicated)**: 11348 rows
- **Duplicate claim_ids**: 681 duplicates (340 pairs)
- **Test (unlabeled)**: 2252 rows
- **Products**: 21 SKUs across 7 families
- **Partners**: 380 partners across 18 cities
- **Train date range**: 2025-04-01 03:24 to 2026-06-30 23:40
- **Test date range**: 2026-07-01 00:41 to 2026-09-30 23:46

## 2. Investigation Outcome Distribution

| Outcome | Count | Percentage |
|---------|-------|------------|
| Not Fraud (0) | 11005 | 96.98% |
| Confirmed Fraud (1) | 141 | 1.24% |
| Undecided (NaN) | 202 | 1.78% |

**Labeled claims**: 11146 (excluding 202 undecided)
**Fraud prevalence**: 141/11146 = **1.27%**
**Class imbalance ratio**: 1:78 (fraud:legitimate)

## 3. Fraud Rate by Product Family

| family            |   Fraud Count |   Total Claims |   Fraud Rate |
|:------------------|--------------:|---------------:|-------------:|
| Robot Vacuum      |            27 |           1621 |   0.0166564  |
| Water Purifier    |            23 |           1548 |   0.0148579  |
| Induction Cooktop |            20 |           1589 |   0.0125865  |
| Air Fryer         |            20 |           1590 |   0.0125786  |
| Mixer Grinder     |            20 |           1603 |   0.0124766  |
| Ceiling Fan       |            16 |           1603 |   0.00998129 |
| Room Heater       |            15 |           1592 |   0.00942211 |

## 4. Fraud Rate by Partner Type

| partner_type              |   Fraud Count |   Total Claims |   Fraud Rate |
|:--------------------------|--------------:|---------------:|-------------:|
| franchise                 |            77 |           3829 |   0.0201097  |
| authorised_service_centre |            52 |           5127 |   0.0101424  |
| freelance_technician      |            12 |           2190 |   0.00547945 |

## 5. Claim Amount Distribution

**Legitimate**: mean=₹2602, median=₹1431, std=₹2979
**Fraud**: mean=₹4522, median=₹2265, std=₹4904

## 6. Warranty Age and Claim Timing

**Legitimate**: mean=236 days, median=215 days
**Fraud**: mean=209 days, median=179 days

## 7. Repeat Claims and Customer Patterns

|   customer_prior_claims |   Fraud Count |   Total |   Fraud Rate |
|------------------------:|--------------:|--------:|-------------:|
|                       0 |            64 |    8192 |    0.0078125 |
|                       1 |            45 |    2501 |    0.0179928 |
|                       2 |            22 |     406 |    0.0541872 |
|                       3 |             6 |      40 |    0.15      |
|                       4 |             2 |       5 |    0.4       |
|                       5 |             1 |       1 |    1         |
|                       6 |             1 |       1 |    1         |

## 8. Photo and Inspection Patterns

**Photo Attached:**
| photo_attached   |   Fraud Count |   Total |   Fraud Rate |
|:-----------------|--------------:|--------:|-------------:|
| N                |            39 |    2448 |    0.0159314 |
| Y                |           102 |    8698 |    0.0117268 |

**Partner Inspected:**
| partner_inspected   |   Fraud Count |   Total |   Fraud Rate |
|:--------------------|--------------:|--------:|-------------:|
| N                   |            45 |    1793 |    0.0250976 |
| Y                   |            96 |    9353 |    0.0102641 |

## 9. Source System Patterns

| source      |   Fraud Count |   Total |   Fraud Rate |
|:------------|--------------:|--------:|-------------:|
| crm         |            95 |    6540 |   0.014526   |
| legacy_zoho |            46 |    4606 |   0.00998697 |

## 10. City Distribution

| city       |   Fraud Count |   Total |   Fraud Rate |
|:-----------|--------------:|--------:|-------------:|
| Pune       |            21 |     494 |   0.0425101  |
| Mysuru     |            16 |     565 |   0.0283186  |
| Bhopal     |            13 |     472 |   0.0275424  |
| Hyderabad  |            14 |     517 |   0.0270793  |
| Hubballi   |            17 |     695 |   0.0244604  |
| Jaipur     |            19 |     931 |   0.0204082  |
| Nashik     |             9 |     552 |   0.0163043  |
| Warangal   |             6 |     571 |   0.0105079  |
| Chennai    |             7 |     738 |   0.00948509 |
| Aurangabad |             5 |     670 |   0.00746269 |
| Coimbatore |             3 |     464 |   0.00646552 |
| Delhi      |             3 |     711 |   0.00421941 |
| Indore     |             2 |     547 |   0.00365631 |
| Nagpur     |             2 |     576 |   0.00347222 |
| Mumbai     |             2 |     729 |   0.00274348 |
| Bengaluru  |             1 |     459 |   0.00217865 |
| Lucknow    |             1 |     679 |   0.00147275 |
| Kota       |             0 |     776 |   0          |

## 11. Partner Tenure Analysis

**Hypothesis test**: Are newer partners more likely to submit fraudulent claims?

**Legitimate**: mean tenure=893 days, median=893 days
**Fraud**: mean tenure=869 days, median=922 days

| tenure_bin   |   Fraud Count |   Total |   Fraud Rate |
|:-------------|--------------:|--------:|-------------:|
| 0-6mo        |            28 |     707 |   0.039604   |
| 6-12mo       |             7 |     902 |   0.00776053 |
| 1-2yr        |            13 |    2580 |   0.00503876 |
| 2-3yr        |            40 |    2999 |   0.0133378  |
| 3yr+         |            53 |    3958 |   0.0133906  |

**Note**: Sample sizes vary substantially across tenure bins. Small samples produce unreliable rates. Partner tenure alone should not be treated as evidence of misconduct.

## 12. Missing Data Summary

### Train:
- `inspector_note`: 3041 (26.8%)
- `is_fraud`: 202 (1.8%)

### Test:
- `inspector_note`: 1829 (81.2%)

## 13. Seasonality and Monthly Volume

| submit_month   |   claims |   fraud |   fraud_rate |
|:---------------|---------:|--------:|-------------:|
| 2025-04        |      739 |       7 |   0.00947226 |
| 2025-05        |      764 |       4 |   0.0052356  |
| 2025-06        |      780 |      11 |   0.0141026  |
| 2025-07        |      793 |      11 |   0.0138714  |
| 2025-08        |      756 |       6 |   0.00793651 |
| 2025-09        |      774 |       7 |   0.00904393 |
| 2025-10        |      736 |       8 |   0.0108696  |
| 2025-11        |      717 |      10 |   0.013947   |
| 2025-12        |      794 |      10 |   0.0125945  |
| 2026-01        |      730 |       8 |   0.0109589  |
| 2026-02        |      694 |       5 |   0.00720461 |
| 2026-03        |      738 |       9 |   0.0121951  |
| 2026-04        |      709 |       9 |   0.0126939  |
| 2026-05        |      709 |      14 |   0.0197461  |
| 2026-06        |      713 |      22 |   0.0308555  |
