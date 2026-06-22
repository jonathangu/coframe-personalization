# Eval Observability Pack

These charts are generated from `solution/results/<dataset>/<policy>.jsonl`.
They explain the walk-forward harness, policy performance, latency, and
serving behavior. Regenerate with:

```bash
uv run python solution/eval_observability.py
```

## policy_scoreboard_heatmap.png

Final captured-headroom table by dataset and policy, including policy averages.

![policy_scoreboard_heatmap](policy_scoreboard_heatmap.png)

## captured_convergence_grid.png

Walk-forward convergence curves showing captured headroom versus training rows.

![captured_convergence_grid](captured_convergence_grid.png)

## final_value_components.png

Policy CVR against the best-fixed baseline and oracle ceiling.

![final_value_components](final_value_components.png)

## policy_runtime.png

Mean train and inference time per policy/window.

![policy_runtime](policy_runtime.png)

## final_recommendation_mix.png

Final-window recommendation distribution by dataset and policy.

![final_recommendation_mix](final_recommendation_mix.png)

## harness_walk_forward_windows.png

Window sizes and cumulative training set sizes used by the harness.

![harness_walk_forward_windows](harness_walk_forward_windows.png)

## choice_violations.png

Invalid action choices caught and replaced by the harness.

![choice_violations](choice_violations.png)
