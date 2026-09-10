# Core V2 NFL Dashboard

Starter Streamlit front end for the Core V2 NFL model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

The app looks for:

`data/week_01_projections.csv`

Until that file is added, it displays the saved NE @ SEA example so the interface can be tested immediately.

## Planned production architecture

1. Model-development notebooks remain separate.
2. A weekly batch pipeline produces one projection file for all games.
3. Streamlit reads only approved production outputs.
4. Market/open/closing lines and actual results are stored separately for grading and CLV tracking.
