# GBIF Spain: can public occurrence data predict where a species shows up next year?

[![CI](https://github.com/AlvaroVerona/gbif-spain-ml-experimentation/actions/workflows/ci.yml/badge.svg)](https://github.com/AlvaroVerona/gbif-spain-ml-experimentation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A small, honest machine-learning experiment on biodiversity data from [GBIF](https://www.gbif.org)
(the Global Biodiversity Information Facility): download species sightings for Spain, check what the
data can and cannot support, and build a first baseline that predicts whether a species will be
**observed in a map grid cell next year**.

The most useful result is a negative one, and it is the point of the project: **with coordinates and
last year's presence alone, the model barely beats chance**, and the exploratory analysis shows why.

> This was the capstone's experimentation stage: a baseline plus a diagnosis of what is missing, not a
> finished forecasting system.

## Key findings

1. **Occurrence data is not movement data.** For 100 well-observed species, the yearly shift of the
   average sighting location has a median of about 3.2° (roughly 350 km). That is far too large to be
   animals migrating; it reflects *where people happened to look* each year. GBIF supports statements
   about changes in the **observed distribution**, not about individual migration.
2. **Accuracy is a misleading metric here.** The first version classified 4,111 of 4,128 test cells
   correctly (99.6%) while finding only 1 of the 11 real positives. Balanced accuracy and ROC AUC are what matter.
3. **The model captures persistence, not ecology.** Its features are coordinates and last year's
   presence. Without climate, land cover or observation effort, there is little else to learn.
4. **Every "absence" is really "not observed".** A predicted absence is not an ecological absence.

## The data

Sightings come from the public GBIF occurrence API (`api.gbif.org/v1/occurrence/search`), country `ES`,
years 2010 to 2025, only records with coordinates, **600 records per year** so that no year dominates.
Only six fields are kept (species, year, month, latitude, longitude, coordinate uncertainty).

| | Documented run (report) | Re-run on 2026-10-01 |
|---|---|---|
| Records | 9,527 | 9,521 |
| Unique species | 1,399 | 1,398 |
| Species with 20+ sightings | 100 | 100 |
| Records after that filter | 6,154 | 6,133 |
| Median yearly centroid shift (degrees) | 3.16 | 3.14 |

The small differences are expected: GBIF is a live database that keeps receiving records, so a download
is a snapshot of that day. Nothing here ships data (parquet files are git-ignored); you download your
own snapshot.

## Model A: next-year presence classifier

- **Task:** for one species, split Spain into grid cells and predict whether the species is observed
  in a cell in year *t+1*, given year *t*.
- **Features:** year, cell-center latitude and longitude, presence in year *t*, presence in year *t-1*.
- **Models:** Logistic Regression and Random Forest, both with balanced class weights.
- **Validation:** a **time-based split**: train on earlier years, test on the last three feature-years.
  Never a random split, because the question is about the future.
- **Version 1** used a 0.5° grid over the whole bounding box: almost every cell is a negative.
  **Version 2** (the committed script) uses 1.0° cells and only cells ever observed for the species,
  which cuts the imbalance.

### Results

Documented in the [Model A report](reports/GBIF_Spain_Model_A_Report.pdf):

| | Test size | Positives | Balanced accuracy | ROC AUC | Positive recall |
|---|---|---|---|---|---|
| v1 (0.5° full mesh) | 4,128 | 11 | 0.545 | 0.705 | 0.09 |
| v2 (1.0° observed cells) | 141 | 11 | 0.568 | n/a in report | 0.18 |

Re-running v2 today (most frequent species, the common blackbird *Turdus merula*, 390 sightings):

| Model | Balanced accuracy | ROC AUC | Positive precision / recall |
|---|---|---|---|
| Logistic Regression | 0.602 | 0.745 | 0.25 / 0.27 |
| Random Forest | 0.549 | 0.579 | 0.15 / 0.18 |

**How much to trust these numbers: not much.** The test set has only 11 positive cases, so the model
gets 2 or 3 of them right; the difference between 0.55 and 0.60 is within noise. The honest conclusion
is the qualitative one above, not a ranking of the two models.

## What I would do next

Taken from the report's own recommendations, in the order I think they matter:

1. **Add environmental drivers** per cell and year (temperature, rainfall, vegetation index, elevation,
   land cover). This is the real bottleneck.
2. **Model observation effort**, or use pseudo-absences, so "not observed" is separated from "not there".
3. **Calibrate the decision threshold** instead of using 0.5, and report precision/recall trade-offs.
4. **Validate more robustly:** rolling-origin time validation and spatial blocking, across many species
   rather than one.
5. **Change the target** to something closer to the scientific question, such as colonization events
   (cell goes from 0 to 1) or latitudinal shift as a regression.

## Run it

```bash
git clone https://github.com/AlvaroVerona/gbif-spain-ml-experimentation.git
cd gbif-spain-ml-experimentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python download_gbif_spain_experiment.py   # ~40 s, writes gbif_spain_experiment.parquet
python gbif_eda_spain.py                   # species counts, yearly centroids, step distances
python ml_presence_spain.py                # trains and evaluates Model A

pip install pytest && pytest               # logic tests on synthetic data, no network
```

Scripts read and write files in the working directory / next to the script, so run them from the
repository root.

## Repository

```text
download_gbif_spain_experiment.py   lightweight dataset used for modelling (6 fields)
download_gbif_spain_full.py         extended raw download (full API objects, 1,000 records/year)
gbif_eda_spain.py                   exploratory analysis: species filter, centroids, step distances
ml_presence_spain.py                Model A (v2): grid, features, time split, models, metrics
tests/test_presence_grid.py         checks the target is next-year presence, the lag uses only the past, the time split has no overlap
reports/                            the three written reports (PDF)
download_gbif.py, gbif_eda.py,
download_gbif_brazil_*.py           earlier Brazil exploration, kept for reference (no report committed)
```

## Limitations

- **One species at a time**, chosen as the most frequent. Conclusions about "birds in Spain" would
  need many species.
- **The sample is balanced by year, not by space or taxon.** The API returns records in its own order,
  so the 600 per year are not a random sample of what was observed.
- **Observation bias dominates.** Popular places and recent years have more sightings, regardless of
  where animals are.
- **Not reproducible to the last row** because GBIF changes daily (see the table above).

## Data and licence

Data: [GBIF.org](https://www.gbif.org), accessed through its public API. Individual records carry their
own licences (CC0, CC BY or CC BY-NC) and the original providers should be credited if you reuse them;
this repository does not redistribute any records. Code: MIT, see [LICENSE](LICENSE).
