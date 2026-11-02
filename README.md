# CabShare Analytics

Static, interactive HTML report built from the Colab cab-sharing analysis and its five CSV datasets. GitHub Pages serves `index.html` from the repository root; the page uses Plotly charts and does not require a Python server for visitors.

## Run locally

From this directory, regenerate the report with:

```powershell
.\.venv\Scripts\python.exe app2.py
```

Or, after installing the dependencies in `requirements.txt`:

```powershell
python app2.py
```

Keep the five CSV datasets beside `app2.py` when regenerating. If they are not there, the generator checks the parent workspace folder. It writes the resulting `index.html` beside the script.

Preview the report locally with:

```powershell
python -m http.server 8000
```

Then open `http://localhost:8000`.

## GitHub Pages

In repository **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/(root)`, and save. GitHub Pages will serve the generated `index.html`.

**Public data notice:** this repository and its Pages site are public. The raw CSVs committed to the repository and the report values/charts are accessible to anyone.

The static report intentionally does not display invented machine-learning scores or train models in a visitor's browser. Re-run the Colab workflow to evaluate and validate model results.

## Project files

- `app2.py` — Python generator for the static report.
- `index.html` — generated report served by GitHub Pages.
- The five CSV files — source datasets used by the report.
- `groupproject.py` — original Colab-exported analysis workflow.
