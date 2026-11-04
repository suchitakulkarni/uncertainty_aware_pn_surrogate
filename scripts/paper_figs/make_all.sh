#!/usr/bin/env bash
# Regenerates every paper figure that is plotted from the tables in paper_data/ (no network needed).
# Outputs: paper_data/sixseed_v10/mirror/figures/*.{png,pdf}.  Run from anywhere.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/../.." MPLBACKEND=Agg TEST_SET=test_fresh250 TEST_RESULTS=test_fresh250_results
export PAPER_FIGS="$PWD/../../paper_data/sixseed_v10/mirror/figures"
mkdir -p "$PAPER_FIGS"
for f in make_fig_coverage_v3b_L1 make_fig_query_answer_v3b_L1 make_fig_eqi_suite_v5_3x3 \
         make_fig_derivative_control_maps_stiffness make_fig_error_map_v3 make_fig_error_map_avgV_pair \
         make_fig_error_map_v3_networkJ_control make_fig_error_map_ensemble_readout_vs_control_jgrad \
         make_fig_iv_overlay make_fig_ablation_distributions make_fig16_cluster sketch2d \
         make_architecture_film_diagram; do
    echo "== $f"; python $f.py "$PAPER_FIGS"
done
