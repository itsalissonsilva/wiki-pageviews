# Experiment plan

## Objective

Forecast the requested horizon for every Wikipedia page while minimizing mean SMAPE. The
first milestone is a trustworthy local validation protocol; model complexity only increases
when it wins across multiple rolling cutoffs and important traffic segments.

## Phases and decision gates

1. **Data audit** — profile missingness, history lengths, intermittency, scale, language,
   access type, and bot/spider agent. Record the dataset fingerprint in MLflow.
2. **Baselines** — evaluate last value, global/rolling median, and seasonal-naive periods 7,
   28, and 364. Promote the best by mean fold SMAPE and inspect per-language failure modes.
3. **Global boosted model** — fit all series jointly on log-transformed traffic using
   calendar, lag, rolling-statistic, trend, and page metadata signals. Forecast recursively.
4. **Optimization** — tune only on inner rolling folds. Use Optuna's TPE sampler with a
   fixed seed. Store every trial as an MLflow child run; refit the winner and score once on
   the final holdout.
5. **Explanation** — log native and permutation importance, grouped importance, residual
   slices, and actual-versus-predicted examples. Treat feature importance as diagnostic,
   not causal evidence.
6. **SOTA benchmark** — compare N-HiTS against the best tuned model on a representative
   cohort, then scale only if the gain justifies cost. Candidate follow-ups are TFT for known
   calendar covariates and an ensemble of seasonal-naive, boosted, and N-HiTS forecasts.

## Evaluation protocol

- Default horizon: 60 days.
- Three expanding-window folds, stepped backward by one horizon.
- Report macro SMAPE (each observation equal), median per-series SMAPE, p90 per-series
  SMAPE, and weighted absolute percentage error as a scale-aware secondary metric.
- Slice results by language, access, agent, volume decile, intermittency, and history length.
- A model advances when mean SMAPE improves and no major slice regresses materially.

## Known risks

- The public challenge test period contains shocks and page-specific events that historical
  lags cannot predict.
- Recursive tree forecasts compound error over long horizons.
- Zero-heavy series make percentage metrics unstable; the implementation uses the official
  zero-denominator convention and reports WAPE alongside SMAPE.
- Full-scale neural training is expensive. Start with a stratified cohort and log wall-clock
  time plus hardware metadata before scaling.

