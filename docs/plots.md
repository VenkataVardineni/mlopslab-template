# DVC Plots Documentation

## Viewing Plots

DVC can generate interactive HTML plots from the forecast data. Use the following commands:

### View plots in browser

```bash
# Generate and open plots HTML
dvc plots show plots/forecast.csv

# Or save to a specific file
dvc plots show plots/forecast.csv -o plots/forecast.html
```

### Compare plots between commits

```bash
# Compare current workspace with HEAD
dvc plots diff

# Compare two specific commits
dvc plots diff HEAD~1 HEAD

# Save comparison to HTML
dvc plots diff HEAD~1 HEAD -o plots/comparison.html
```

## Plot Data Format

The forecast plot data (`plots/forecast.csv`) contains:
- `timestamp`: Time index for the forecast
- `actual`: Actual observed values
- `pred_p50`: Median forecast (50th percentile)
- `pred_p10`: Lower bound forecast (10th percentile)
- `pred_p90`: Upper bound forecast (90th percentile)

## Customizing Plots

You can customize the plot appearance by modifying the `dvc.yaml` plots configuration:

```yaml
plots:
  - plots/forecast.csv:
      x: timestamp
      y:
        - actual
        - pred_p50
        - pred_p10
        - pred_p90
      template: linear
```

Available templates:
- `linear`: Line plot (default)
- `scatter`: Scatter plot
- `smooth`: Smooth line plot

