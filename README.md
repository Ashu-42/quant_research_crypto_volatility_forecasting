# Crypto Volatility Forecasting

This repository contains the code and notebooks for a quantitative
research project on cryptocurrency return and volatility forecasting.

## Current assets

- Bitcoin — BTCUSDT
- Ethereum — ETHUSDT
- Solana — SOLUSDT
- XRP — XRPUSDT

## Data source

Binance public Spot market-data API.

## Current dataset

- Frequency: 15-minute candles
- Requested start date: 2016-07-01
- End date: configurable
- Actual history depends on the listing date of each trading pair

Large datasets are stored separately in a shared Google Drive folder
and are not committed to this repository.

## Current pipeline

1. Pull paginated Binance candlestick data.
2. Convert timestamps and numerical columns.
3. Calculate returns, intraday range, and volume changes.
4. Validate date ranges, duplicates, and missing values.
5. Save processed data as Parquet.
6. Share datasets through Google Drive.

## Repository structure

```text
notebooks/
    Research and development notebooks

src/
    Reusable Python modules - TBA

configs/
    Project configuration files - TBA

tests/
    Data and code validation tests - TBA
