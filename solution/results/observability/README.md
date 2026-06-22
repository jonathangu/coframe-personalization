# Data Observability Pack

These displays summarize the logged data used by the policy. They are
generated from the observed logs, not from the hidden truth files.

## Displays

### overview.png

Dataset size, action count, observed reward rate, and whether propensities are logged.

![overview](overview.png)

### assignment_mix.png

Overall logged action mix by dataset.

![assignment_mix](assignment_mix.png)

### reward_by_variant_support.png

Observed reward by action with binomial uncertainty; red diamonds show IPW means where propensities exist.

![reward_by_variant_support](reward_by_variant_support.png)

### propensity_diagnostics.png

Propensity distributions and effective sample size diagnostics.

![propensity_diagnostics](propensity_diagnostics.png)

### action_availability.png

First-to-last appearance of each action over the month.

![action_availability](action_availability.png)

### temporal_assignment_mix.png

How assignment mix changes across chronological bins.

![temporal_assignment_mix](temporal_assignment_mix.png)

### temporal_reward.png

Observed conversion rate over chronological bins.

![temporal_reward](temporal_reward.png)

### segment_signal_heatmap.png

Single-feature smoothed segment signal proxy used to guide policy design.

![segment_signal_heatmap](segment_signal_heatmap.png)
