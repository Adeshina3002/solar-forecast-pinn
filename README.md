# Physics-Informed Solar Forecasting

> Hour-ahead solar generation forecasting for the German grid, combining a
> physics-informed neural network with classical ML baselines. Trained on
> Germany; out-of-distribution validation on Italy.

**Status:** 🚧 In development — Phase 1: data ingestion and quality profiling.

## Why this exists

Pure ML solar forecasters learn the physics of photovoltaic generation from
data alone. They work, but they are sample-hungry, struggle on edge cases
(snow, low-sun winter days, clipping events), and have no way to express what
we already know from first principles — that PV output is governed by an
irradiance-temperature equation that's been on the back of every solar
engineer's envelope for 40 years.

This project asks: **does injecting that physical prior as a soft constraint
in the loss function beat a tuned XGBoost baseline?** And: **does the prior
transfer to a different climate (Italy) better than the baseline does?**

## Data

Two pinned packages from the [Open Power System Data](https://open-power-system-data.org)
platform:

- **Time series** (`v2020-10-06`): hourly load and solar generation for European
  countries, with per-TSO breakdown for Germany. 50,401 rows × ~400 columns,
  Dec 2014 → Sep 2020.
- **Weather** (`v2020-09-16`): hourly temperature and irradiance for the same
  countries, derived from NASA MERRA-2 reanalysis. 350,640 rows × 67 columns,
  1980 → 2019.

Download with:
```bash
python scripts/download_data.py
```

Data files are gitignored. The script verifies SHA-256 against OPSD's published
manifest so the dataset is bit-identical across machines.

## Architecture (planned)

1. **Baseline 1**: Seasonal naive + hour-of-day median (the "must beat" floor)
2. **Baseline 2**: XGBoost with engineered weather features
3. **Pure ML**: LSTM with weather + lagged generation
4. **Physics-Informed NN**: same LSTM, plus a soft penalty in the loss that
   enforces the PV power equation `P = η·A·G_POA·[1 − β(T_cell − T_ref)]`
5. **Hybrid**: PINN residual on top of an analytical physics base estimate

Each stage adds one idea at a time so the contribution of the physics prior is
isolable.

## Repo layout

```
solar-forecast-pinn/
├── scripts/
│   ├── download_data.py        # fetch OPSD packages
│   └── data_quality_report.py  # missingness, coverage, sanity checks
├── src/
│   └── solar_forecast/         # importable package (added in week 2)
├── notebooks/                  # exploration, not production code
├── tests/                      # pytest suite
├── data/                       # gitignored
└── models/                     # gitignored
```

## Setup

```bash
git clone <repo-url>
cd solar-forecast-pinn
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Author

Oladeji Adeshina — physicist (MSc UNICAL) and backend engineer.
[@dev_virtuoso](https://x.com/dev_virtuoso) · [dev-virtuoso.com](https://dev-virtuoso.com)

## License

MIT
