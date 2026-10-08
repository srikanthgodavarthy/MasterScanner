# SMC Evidence Funnel diagnostic — READ-ONLY, manual trigger only.
# Runs scripts/smc_funnel_diagnostic.py on GitHub's runners (no local setup) and
# prints the report on the run's Summary page. Touches no database, no secrets,
# and nothing in the scoring path.
name: SMC funnel diagnostic

on:
  workflow_dispatch:
    inputs:
      limit:
        description: "Max symbols to fetch (0 = whole Nifty 500)"
        default: "150"
      days:
        description: "Evaluation window in bars"
        default: "60"
      sleep:
        description: "Seconds between Yahoo fetches (raise if many fail)"
        default: "0.3"

jobs:
  funnel:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Run SMC funnel diagnostic
        run: |
          python scripts/smc_funnel_diagnostic.py \
            --universe nifty500 \
            --limit "${{ inputs.limit }}" \
            --days "${{ inputs.days }}" \
            --sleep "${{ inputs.sleep }}" \
            --sensitivity \
            --out smc_funnel_out
          cat smc_funnel_out/funnel_summary.md >> "$GITHUB_STEP_SUMMARY"
      - uses: actions/upload-artifact@v4
        with:
          name: smc-funnel
          path: smc_funnel_out/
