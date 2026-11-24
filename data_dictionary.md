# Data Dictionary — Kestrel Warranty Fraud Detection

## Claims Data (train.csv / test_unlabelled.csv)

| Column | Type | Description | Available at Prediction Time? |
|--------|------|-------------|------------------------------|
| `claim_id` | string | Unique claim number (e.g. WC700000). Partners may re-submit bounced claims, so the same claim_id can appear twice in train. | Yes — identifier |
| `submitted_at` | datetime string (IST) | Timestamp when claim was submitted, format `YYYY-MM-DD HH:MM`. CRM records are IST; legacy Zoho resolution events were UTC (not converted during migration). | Yes |
| `partner_id` | string | Service partner identifier (e.g. SP3150), foreign key to partners.csv | Yes |
| `sku` | string | Product SKU (e.g. KH-RV-03), foreign key to products.csv | Yes |
| `product_serial` | string | Serial number as typed by the partner — expect data-entry errors, inconsistent casing, possible hyphens | Yes |
| `days_since_purchase` | int | Days between original purchase date and claim submission | Yes |
| `claim_amount_inr` | float | Amount claimed in INR. Range: ₹250–₹26,728 | Yes |
| `photo_attached` | string (Y/N) | Whether the partner attached photos | Yes |
| `partner_inspected` | string (Y/N) | Whether the partner signed off on inspection. Policy: required until 30 Apr 2026; from 1 May 2026, claims <₹2,000 auto-approved without inspection | Yes |
| `claim_description` | string | Free-text fault description written by the partner | Yes |
| `inspector_note` | string | Inspector's note (free text). Blank/NaN if not inspected. **LEAKAGE RISK** — this is populated after inspection, potentially after claim submission | Partial — blank for uninspected claims; content reveals investigation outcome |
| `customer_prior_claims` | int | Count of earlier warranty claims by the same customer. Range: 0–6 | Yes |
| `source` | string | System of record: `crm` (from 1 Oct 2025) or `legacy_zoho` (before 1 Oct 2025) | Yes |
| `is_fraud` | float | **TARGET** (train only). 1 = confirmed fraud, 0 = not fraud, NaN = undecided/still under investigation at export time. Per email: "Zoho couldn't store blanks, so the old ones came across as 0" — meaning some legacy_zoho 0s may actually be undecided | Train only |

## Products Reference (products.csv)

| Column | Type | Description |
|--------|------|-------------|
| `sku` | string | Product SKU (primary key, 21 unique) |
| `family` | string | Product category: Air Fryer, Mixer Grinder, Water Purifier, Robot Vacuum, Induction Cooktop, Ceiling Fan, Room Heater |
| `list_price_inr` | int | Retail list price in INR |
| `warranty_months` | int | Standard warranty period: 12 or 24 months. Mixer Grinders and Ceiling Fans get 24 months; all others 12 months |

## Partners Reference (partners.csv)

| Column | Type | Description |
|--------|------|-------------|
| `partner_id` | string | Service partner identifier (primary key, 380 unique) |
| `city` | string | City where partner operates (17 unique cities) |
| `onboarded_date` | date string | Date the partner was onboarded (YYYY-MM-DD) |
| `partner_type` | string | Category: `authorised_service_centre`, `franchise`, `freelance_technician` |

## Key Relationships
- `claims.partner_id` → `partners.partner_id` (many-to-one)
- `claims.sku` → `products.sku` (many-to-one)
- Sample submission expects: `claim_id, score` for all 2,252 test rows

## Data Quality Notes
- **Duplicate claim_ids in train**: 1,362 rows have duplicate claim_ids (681 pairs). Tanmay confirms partners re-submit bounced claims.
- **inspector_note missing**: 3,237 of 12,029 train rows; 1,829 of 2,252 test rows
- **is_fraud NaN**: 215 rows in train (all from CRM source) — undecided investigations
- **Legacy Zoho 0s**: Some `is_fraud=0` from legacy_zoho may actually be undecided cases that Zoho couldn't store as blank
- **Serial number inconsistencies**: Mixed casing, some with hyphens, some without
