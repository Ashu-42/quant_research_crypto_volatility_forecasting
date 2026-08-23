# Intraday Cryptocurrency Volatility Forecasting — 15-Minute Horizon

## Project overview

This repository contains the **15-minute intraday branch** of a cryptocurrency volatility-forecasting research project.

The central research question is:

> **Can deep-learning models using intraday market state, long-memory volatility information, and recent temporal sequences improve one-step-ahead crypto volatility forecasts relative to classical ARCH/GARCH models and simple historical-volatility baselines?**

The final study covers four highly liquid crypto assets:

- **Bitcoin (BTCUSDT)**
- **Ethereum (ETHUSDT)**
- **Solana (SOLUSDT)**
- **XRP (XRPUSDT)**

The final modeling universe includes:

- simple statistical / historical-volatility baselines;
- GARCH-family econometric models;
- MLP models;
- GRU models;
- LSTM models;
- asset-specific and pooled/global deep-learning variants.

The final held-out Test evaluation is now complete.

---

# 1. Research evolution

The project began with a lower-frequency daily volatility setup. The intraday branch was introduced to test whether a richer 15-minute dataset could provide:

1. substantially more observations;
2. explicit intraday volatility-seasonality information;
3. richer range/activity features;
4. recent local sequences for recurrent models;
5. improved comparison between classical volatility models and modern deep learning.

The 15-minute branch ultimately became the main empirical experiment because it produced a much larger supervised sample and allowed the neural models to use information that is unavailable or much weaker at daily frequency.

The final approach deliberately separates:

- **econometric conditional-volatility models**;
- **engineered-state neural models**;
- **sequence neural models**;
- **simple historical-volatility benchmarks**.

This makes the comparison useful not only for identifying the best forecast but also for understanding **which type of volatility information adds value**.

---

# 2. Forecasting problem

## Forecast horizon

Every model forecasts **one 15-minute candle ahead**.

If the current candle is indexed by \(t\), the forecast target is associated with candle \(t+1\).

## Return definition

Close-to-close log return:

\[
r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)
\]

## Primary variance target

The realized-variance proxy is:

\[
RV_t = r_t^2
\]

and therefore the forecasting target is:

\[
RV_{t+1}=r_{t+1}^2
\]

This is kept consistent across the baseline, ARCH and deep-learning models.

## Primary evaluation metric

**QLIKE** is the primary forecast-loss metric.

Supporting metrics:

- MAE
- RMSE

QLIKE is emphasized because volatility is a latent quantity and squared return is a noisy proxy. QLIKE is widely used for volatility forecast comparison and penalizes badly calibrated variance forecasts more appropriately than simply minimizing squared error.

---

# 3. Data splits

The split is chronological and fixed.

| Split | Target period |
|---|---|
| **Train** | Up to 2024-07-31 23:45 UTC |
| **Validation / development** | 2024-08-01 00:00 UTC → 2025-07-31 23:45 UTC |
| **Test** | 2025-08-01 00:00 UTC → 2026-07-31 23:45 UTC |

At 15-minute frequency:

- Validation contains **35,040 target candles per asset**
- Test contains **35,040 target candles per asset**

The final holistic notebook confirmed complete canonical Test coverage for all four assets.

The common Train samples used in the final all-model comparison contain approximately:

| Asset | Train observations |
|---|---:|
| BTC | 238,464 |
| ETH | 238,464 |
| SOL | 137,589 |
| XRP | 214,702 |

Train starts differ because the assets have different historical-data availability.

### Important research-process note

Validation is treated as a **development/model-selection set**. The intraday methodology evolved while the project was being developed, so Validation should not be described as a completely untouched final holdout.

The **Test period is the true final frozen holdout**.

Before Test was opened, the final comparison notebook reproduced the previously saved Validation forecasts for all model families and required the reproduction gate to pass.

Only after that gate passed was Test evaluated.

No model should be retuned after viewing the Test results.

---

# 4. Data integrity and leakage controls

The project uses strict safeguards because leakage can easily occur in high-frequency volatility forecasting.

Key controls include:

- one-candle-ahead target alignment;
- target timestamps explicitly separated from current-candle timestamps;
- sequences must be chronologically contiguous;
- sequences cannot cross missing-candle gaps;
- non-canonical gap-crossing rows cannot become supervised targets;
- feature scalers are fit on Train only;
- intraday seasonality factors are estimated on Train only;
- inner hyperparameter tuning occurs within Train;
- Test is never used for feature selection, lookback selection, learning-rate selection or epoch selection;
- target realized variance, target return and other realized target quantities cannot enter model inputs;
- only **known-ahead target calendar variables** such as time-of-day and day-of-week encodings are allowed;
- global neural models use per-asset Train-only scaling;
- all final model comparisons use identical target timestamps inside each asset/split.

The final holistic evaluation also performs a **saved Validation forecast reproduction test**.

All saved Validation forecasts were reproduced successfully before Test evaluation.

---

# 5. Intraday statistical properties

A dedicated Train-only EDA/statistical-diagnostics notebook was used to avoid learning from the final holdout.

The intraday returns show the standard stylized facts expected from financial volatility data, but very strongly.

## Heavy tails

Excess kurtosis is large for every asset:

| Asset | Excess kurtosis |
|---|---:|
| BTC | ~66.2 |
| ETH | ~50.5 |
| SOL | ~29.5 |
| XRP | ~138.8 |

Jarque–Bera strongly rejects Normality across the assets.

Student-t fits also imply very heavy tails.

## Return stationarity

ADF and Phillips–Perron diagnostics reject a unit root in returns.

Raw returns therefore behave approximately as stationary series, while volatility transforms such as log-squared returns exhibit much stronger persistence / possible regime variation.

## Volatility clustering

Return autocorrelation is economically small, but:

- absolute-return autocorrelation is persistent;
- squared-return autocorrelation is persistent;
- Ljung–Box rejects independence for volatility transforms;
- ARCH-LM strongly rejects the null of no ARCH effect.

This provides direct empirical motivation for conditional-volatility modeling.

## Intraday seasonality

Volatility varies materially by intraday clock slot.

The deep-learning pipeline therefore estimates **training-only time-of-day variance factors** and predicts deseasonalized variance before restoring the forecast to the original variance scale.

## Cross-asset dependence

The assets also exhibit substantial contemporaneous dependence.

For example, BTC–ETH Train-period correlations were roughly:

- return correlation: **~0.85**
- squared-return correlation: **~0.76**

This is one reason pooled/global neural models were included rather than restricting the study to independent asset-specific models.

---

# 6. Feature engineering

The final deep-learning models use a richer information set than raw returns alone.

The exact features are generated dynamically from the configured candle interval.

Major feature groups include:

## Volatility state

- current realized variance;
- deseasonalized current variance;
- log adjusted variance;
- multiple rolling realized-variance means;
- lagged volatility states;
- short-horizon / long-horizon volatility ratios;
- HAR-style multi-scale state variables.

## Return state

- current return;
- absolute return;
- deseasonalized return;
- intrabar open-to-close return.

## Range-based volatility

- Parkinson variance;
- Garman–Klass variance;
- current and rolling range-volatility states.

## Asymmetric volatility

- positive semivariance;
- negative semivariance;
- rolling positive/negative semivariance states.

## Market activity

- quote volume;
- number of trades;
- taker-buy imbalance;
- average trade size;
- rolling activity surprises.

## Known-ahead calendar state

- target time-of-day sine/cosine;
- target day-of-week sine/cosine.

These target-calendar variables are known before the target candle occurs and therefore do not constitute leakage.

---

# 7. Intraday seasonality adjustment

Raw high-frequency crypto volatility has a strong clock-time pattern.

For the final neural models:

1. variance seasonality is estimated using Train observations only;
2. current variance is divided by the current-slot variance factor;
3. target variance is divided by the target-slot factor;
4. the model predicts **log adjusted variance**;
5. the prediction is exponentiated;
6. target-slot seasonality is multiplied back in;
7. QLIKE / MAE / RMSE are calculated on the original variance scale.

This allows the model to learn deviations from predictable intraday seasonality rather than spending model capacity relearning the same clock pattern.

---

# 8. Models

## 8.1 Statistical / historical-volatility baselines

The final comparison includes:

- Persistence
- Mean RV (4h)
- Mean RV (1d)
- Same Slot Previous Day
- Train Slot Mean RV
- Rolling Std (1d, causal)

### Causal rolling standard deviation

The causal rolling-standard-deviation baseline uses historical returns only through forecast origin \(t\).

It is therefore a genuine forecasting baseline.

### Ex-post rolling standard deviation

Separate 4h / 12h / 1d rolling return-standard-deviation series are also constructed for descriptive comparison.

These include the realized target period and are therefore **not predictive benchmarks**. They are used only to visualize how model-implied volatility tracks realized historical volatility.

This distinction is important for interpretation.

---

## 8.2 ARCH-family models

Three standard econometric models are evaluated:

- GARCH(1,1)
- GJR-GARCH(1,1)
- EGARCH(1,1)

The final Test notebook refits the ARCH models using only information available before each forecast split.

---

## 8.3 Final MLP

The final MLP uses the rich engineered volatility/activity state directly.

### Configuration

- feature set: **FULL_ACTIVITY**
- state features: **48**
- learning rate: **3e-4**
- QLIKE-aligned Gaussian variance loss

Both:

- Asset-Specific MLP
- Global MLP

are retained.

The Global MLP uses pooled information across BTC, ETH, SOL and XRP while maintaining asset-specific preprocessing.

---

## 8.4 Final GRU

The GRU adds an explicitly ordered local sequence to the same long-memory volatility-state information.

### Architecture

Two branches:

1. recent causal GRU sequence branch;
2. engineered multi-scale volatility-state branch.

The branches are fused before the final log-variance forecast.

### Selected configuration

- sequence lookback: **4h**
- 15-minute bars: **16**
- learning rate: **3e-4**
- sequence features: **12**
- state features: **~48**

Both Asset-Specific GRU and Global GRU are retained.

The Global model also incorporates asset identity and asset-balanced training weights.

---

## 8.5 Final LSTM

The LSTM was designed as the longest-memory neural sequence benchmark.

It uses:

- a forward-only LSTM;
- final hidden state;
- temporal attention over observed hidden states;
- the same multi-scale engineered state branch;
- QLIKE-aligned variance training.

### Selected configuration

- sequence lookback: **12h**
- 15-minute bars: **48**
- learning rate: **3e-4**
- sequence features: **12**
- state features: **48**
- final Global LSTM epochs: **21**

Asset-specific selected epochs:

| Asset | Epochs |
|---|---:|
| BTC | 20 |
| ETH | 10 |
| SOL | 19 |
| XRP | 7 |

Both Asset-Specific LSTM and Global LSTM are retained.

---

# 9. Validation results

QLIKE is lower-is-better.

## Validation winner by asset

| Asset | Best Validation model | QLIKE |
|---|---|---:|
| BTC | **Asset-Specific LSTM** | **1.436966** |
| ETH | **Global LSTM** | **1.488880** |
| SOL | **Asset-Specific MLP** | **1.418898** |
| XRP | **Global GRU** | **1.557016** |

The best ARCH Validation QLIKE values were:

| Asset | Best ARCH | QLIKE |
|---|---|---:|
| BTC | EGARCH(1,1) | 1.515321 |
| ETH | EGARCH(1,1) | 1.548225 |
| SOL | EGARCH(1,1) | 1.469180 |
| XRP | GARCH(1,1) | 1.632367 |

Thus the enhanced deep-learning models beat the best ARCH-family Validation forecast for all four assets.

### Cross-asset mean Validation QLIKE

Selected single-model averages across the four assets:

| Model | Mean Validation QLIKE |
|---|---:|
| **Global GRU** | **1.482251** |
| Global LSTM | 1.482852 |
| Asset-Specific MLP | 1.485252 |
| Asset-Specific GRU | 1.486436 |
| Asset-Specific LSTM | 1.489407 |
| Global MLP | 1.493855 |
| EGARCH(1,1) | 1.547919 |

Global GRU and Global LSTM are extremely close on average during Validation.

---

# 10. Final held-out Test results

The Test period was opened only after:

1. every model was frozen;
2. saved Validation forecasts were reproduced;
3. common-date QC passed.

QLIKE remains the primary ranking metric.

## Overall Test winner among the frozen model set

| Asset | Best Test model | Test QLIKE | Best ARCH | ARCH QLIKE | QLIKE improvement vs best ARCH |
|---|---|---:|---|---:|---:|
| BTC | **Asset-Specific GRU** | **1.598606** | EGARCH(1,1) | 1.666965 | **4.10%** |
| ETH | **Asset-Specific GRU** | **1.667876** | EGARCH(1,1) | 1.728975 | **3.53%** |
| SOL | **Asset-Specific MLP** | **1.674185** | EGARCH(1,1) | 1.711059 | **2.16%** |
| XRP | **Global LSTM** | **1.620906** | EGARCH(1,1) | 1.649036 | **1.71%** |

These “best Test model” labels are **descriptive rankings of already-frozen models**, not models selected or retuned using Test.

Across the four assets, the average QLIKE of the best frozen deep-learning forecast is approximately **1.6404**, compared with approximately **1.6890** for the best ARCH model in each asset.

That is about a **2.9% lower mean QLIKE**.

---

# 11. Complete primary-model Test QLIKE comparison

| Model | BTC | ETH | SOL | XRP | Cross-asset mean |
|---|---:|---:|---:|---:|---:|
| **Global GRU** | 1.610477 | 1.679831 | 1.685032 | 1.623372 | **1.649678** |
| Asset-Specific GRU | **1.598606** | **1.667876** | 1.686729 | 1.668926 | 1.655534 |
| Asset-Specific LSTM | 1.601562 | 1.676345 | 1.712041 | 1.651627 | 1.660394 |
| Asset-Specific MLP | 1.651662 | 1.688418 | **1.674185** | 1.632382 | 1.661662 |
| Global LSTM | 1.652677 | 1.705747 | 1.679785 | **1.620906** | 1.664779 |
| Global MLP | 1.610084 | 1.728261 | 1.726638 | 1.662189 | 1.681793 |
| **EGARCH(1,1)** | 1.666965 | 1.728975 | 1.711059 | 1.649036 | **1.689009** |
| GARCH(1,1) | 1.689639 | 1.776547 | 1.790945 | 1.659312 | 1.729111 |
| GJR-GARCH(1,1) | 2.831208 | 1.755101 | 1.767407 | 1.695535 | 2.012313 |

### Best single model across all four assets

If one common model must be chosen for the entire four-asset universe, **Global GRU has the lowest average held-out Test QLIKE: ~1.6497**.

This is an important distinction from choosing a different ex-post winner for each asset.

---

# 12. Validation vs Test behavior

The final results show that there is **no universal architecture winner**.

Validation winners and Test rankings do not always coincide:

| Asset | Validation winner | Test rank of that model |
|---|---|---:|
| BTC | Asset-Specific LSTM | 2 |
| ETH | Global LSTM | 5 |
| SOL | Asset-Specific MLP | 1 |
| XRP | Global GRU | 2 |

This is an important empirical result rather than a failure of the experiment.

It suggests:

- crypto volatility relationships are regime-dependent;
- small Validation differences between neural models should not be overinterpreted;
- pooled/global learning can help some assets substantially;
- sequence models are valuable, but longer memory is not always better;
- engineered state alone remains highly competitive for SOL.

In particular:

- GRU is strongest on BTC and ETH Test QLIKE;
- MLP remains strongest for SOL;
- Global LSTM is strongest for XRP;
- Global GRU is the strongest single architecture on average across the full four-asset Test set.

---

# 13. Train / Validation / Test diagnostics

The final holistic notebook computes MAE, RMSE and QLIKE for every model on:

- Train
- Validation
- Test

and saves both long-form and wide-form outputs.

These are intended for **generalization diagnostics**, not for post-Test tuning.

Example QLIKE trajectory for the Validation-winning model in each asset:

| Asset | Model | Train | Validation | Test |
|---|---|---:|---:|---:|
| BTC | Asset-Specific LSTM | 1.603808 | 1.436966 | 1.601562 |
| ETH | Global LSTM | 1.608026 | 1.488880 | 1.705747 |
| SOL | Asset-Specific MLP | 1.718546 | 1.418898 | 1.674185 |
| XRP | Global GRU | 1.762168 | 1.557016 | 1.623372 |

Train error is not expected to be mechanically below Validation/Test error because the Train period is much longer and spans a different volatility regime.

The main generalization comparison is therefore the transition from the development Validation period to the frozen Test period.

---

# 14. Historical-volatility comparison

The final notebook includes two different rolling-volatility concepts.

## Predictive rolling-std baseline

`Rolling Std (1d, causal)`

uses only returns known before the target candle.

Test QLIKE:

| Asset | Rolling Std (1d, causal) |
|---|---:|
| BTC | 1.851687 |
| ETH | 1.926555 |
| SOL | 1.854372 |
| XRP | 1.804659 |

All leading neural and ARCH models outperform this simple historical-volatility baseline on QLIKE.

## Ex-post rolling-volatility reference

The final visualization also compares predicted volatility with rolling return standard deviation over configurable durations such as:

- 4h
- 12h
- 1d

This ex-post series includes realized target information and is **not** included as a predictive competitor.

It is used only to assess whether forecasts move with the realized volatility regime.

---

# 15. Main findings

The final 15-minute experiment supports several conclusions.

## 1. Deep learning improves QLIKE relative to ARCH

The strongest frozen deep-learning model beats the strongest ARCH forecast on held-out Test for **all four assets**.

The advantage is meaningful but not enormous: approximately **1.7%–4.1%** lower QLIKE by asset.

This is more credible than claiming overwhelming superiority.

## 2. Sequence information adds value

GRU is the Test winner for BTC and ETH, suggesting that recent temporal ordering provides information beyond the engineered state alone.

## 3. Longer sequence memory is not universally superior

LSTM uses a substantially longer selected memory window than GRU:

- GRU: 4h / 16 bars
- LSTM: 12h / 48 bars

Yet LSTM is not consistently better.

This implies that useful crypto volatility memory is asset/regime dependent and that more recurrent depth/history does not automatically improve forecasts.

## 4. The MLP remains extremely competitive

Asset-Specific MLP is the best Test model for SOL.

This shows that rich heterogeneous state features can contain most of the relevant information without requiring a recurrent architecture for every asset.

## 5. Global training can be valuable

Global GRU is the **best single model by mean Test QLIKE across all four assets**.

Global LSTM is the best XRP model.

Cross-asset pooling therefore provides useful information, particularly where individual-asset histories are shorter or noisier.

## 6. Intraday seasonality matters

Strong time-of-day volatility patterns justify the Train-only deseasonalization used by the deep-learning pipeline.

## 7. Classical volatility models remain strong benchmarks

EGARCH is consistently competitive and is the strongest ARCH model on Test for all four assets.

The final result should therefore be framed as an incremental but consistent improvement over a strong econometric benchmark.

---

# 16. Notebook workflow

Recommended order for the intraday branch:

```text
02_Intraday_Data_Processing_and_EDA.ipynb
        ↓
02B_Intraday_EDA_Statistical_Tests.ipynb
        ↓
03_Intraday_Baselines_and_ARCH.ipynb
        ↓
04_Intraday_MLP_FINAL.ipynb
        ↓
05_Intraday_GRU_FINAL_CORRECTED.ipynb
        ↓
06_Intraday_LSTM_FINAL.ipynb
        ↓
06B_LSTM_Recovery_Global_Final_and_Comparison*.ipynb
        ↓
07_Intraday_Final_Holistic_Comparison_Train_Validation_Test.ipynb
```

### Note on Notebook 06B

`06B` is a recovery notebook created after a Colab runtime disconnected during the final Global LSTM fit.

It:

- reused already-saved asset-specific LSTM artifacts;
- reused the frozen LSTM configuration;
- trained only the missing final Global LSTM;
- did not retune LSTM hyperparameters.

For a fresh clean reproduction where Notebook 06 completes normally, Notebook 06B is not conceptually required.

---

# 17. Final evaluation protocol

Notebook 07 is the final evaluation notebook.

Its sequence is deliberately conservative:

1. discover the configured interval from the processing manifest;
2. load every frozen model and preprocessing artifact;
3. dynamically reconstruct required features/sequences;
4. reconstruct common Train / Validation / Test timestamps;
5. regenerate Validation forecasts;
6. compare regenerated Validation predictions with the original saved predictions;
7. require the Validation reproduction gate to pass;
8. only then open the held-out Test set;
9. evaluate every model on identical Test timestamps;
10. save QLIKE / MAE / RMSE tables, rankings and figures;
11. mark post-Test tuning as disallowed.

The executed final run reported:

```text
ALL SAVED VALIDATION FORECASTS REPRODUCED SUCCESSFULLY.
The frozen pipeline is consistent. Test may now be opened.

Test has complete canonical coverage.

ALL FINAL INTRADAY HOLISTIC COMPARISON QC CHECKS PASSED.
Validation reproduction gate: PASSED
Held-out Test: EVALUATED
```

---

# 18. Final result artifacts

The final comparison notebook saves to:

```text
results/<interval>/FINAL_HOLISTIC_COMPARISON/
```

including:

```text
all_models_train_validation_test_metrics_long.csv
common_evaluation_date_qc.csv

QLIKE_train_validation_test.csv
RMSE_train_validation_test.csv
MAE_train_validation_test.csv

validation_ranking_all_models.csv
test_ranking_all_models.csv
generalization_gap_metrics.csv

rolling_std_proxy_test_metrics.csv
test_predictions_all_models.parquet

arch_refit_diagnostics.csv
final_evaluation_manifest.json

figures/
```

These should be the primary source tables for the paper.

---

# 19. Suggested repository organization

A lightweight repository structure is recommended:

```text
.
├── README.md
├── notebooks/
│   ├── 02_Intraday_Data_Processing_and_EDA.ipynb
│   ├── 02B_Intraday_EDA_Statistical_Tests.ipynb
│   ├── 03_Intraday_Baselines_and_ARCH.ipynb
│   ├── 04_Intraday_MLP_FINAL.ipynb
│   ├── 05_Intraday_GRU_FINAL_CORRECTED.ipynb
│   ├── 06_Intraday_LSTM_FINAL.ipynb
│   ├── 06B_LSTM_Recovery_Global_Final_and_Comparison*.ipynb
│   └── 07_Intraday_Final_Holistic_Comparison_Train_Validation_Test.ipynb
├── results/
│   └── lightweight final CSV summaries
├── docs/
│   └── paper notes / experiment documentation
└── .gitignore
```

Large raw parquet files, TensorFlow models and intermediate artifacts should generally remain outside Git and be referenced through the project storage structure.

---

# 20. Runtime / environment

The final holistic evaluation was executed with:

- TensorFlow **2.20.0**
- `arch` **7.2.0**

The notebooks additionally rely on common scientific Python packages including:

- NumPy
- pandas
- SciPy
- statsmodels
- scikit-learn
- joblib
- Matplotlib
- PyArrow / parquet support

GPU runtime is strongly recommended for GRU/LSTM training.

---

# 21. Reproducibility principles

To reproduce the reported results:

1. preserve the chronological split boundaries;
2. do not estimate seasonality using Validation/Test;
3. fit all scalers on Train only;
4. retain one-candle-ahead target alignment;
5. retain canonical-gap filtering;
6. do not change model hyperparameters after Test has been viewed;
7. compare all models on identical timestamps;
8. use QLIKE as the primary model-comparison metric;
9. treat ex-post rolling standard deviation as visualization only;
10. retain the final evaluation manifest with all published results.

---

# 22. Interpretation for a research paper

The strongest defensible conclusion from this experiment is not that one neural architecture dominates volatility forecasting universally.

A more accurate conclusion is:

> **At a 15-minute crypto forecasting horizon, neural models combining intraday seasonality adjustment, heterogeneous multi-scale volatility state, market-activity information and, where useful, recent sequential information consistently improve QLIKE relative to standard GARCH-family forecasts. However, the preferred neural architecture varies by asset and period, and a pooled GRU provides the strongest average held-out performance across the four assets.**

This result is interesting because it combines three observations:

1. classical EGARCH remains a strong benchmark;
2. richer neural volatility-state representations produce consistent incremental gains;
3. sequence length and architecture choice remain asset/regime dependent.

The paper should therefore emphasize:

- forecast robustness;
- fair temporal evaluation;
- cross-asset generalization;
- statistical/economic interpretation of the improvement;
- the absence of a universal model winner.

---

# 23. Current project status

**15-minute branch: modeling complete.**

Completed:

- data processing;
- Train-only EDA and statistical testing;
- baseline models;
- GARCH/GJR/EGARCH;
- MLP;
- GRU;
- LSTM;
- asset-specific/global comparison;
- Validation reproduction gate;
- final held-out Test evaluation;
- rolling historical-volatility comparison;
- Train/Validation/Test error comparison;
- final model rankings and exported artifacts.

The Test set has now been evaluated.

**No further hyperparameter tuning should be performed on the models reported here.**

Future work should be treated as a new experiment with a new holdout design rather than an extension of the current Test-tuned pipeline.
