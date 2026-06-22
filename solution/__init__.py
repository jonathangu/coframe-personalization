from engine.policy import BestFixedVariantPolicy
from policies import register

from solution.eb_policy import AdditiveEBPolicy
from solution.feature_demo import TimeOfDayPolicy
from solution.gbm_policy import GBMTLearnerPolicy, HybridEBGBMPolicy
from solution.lgbm_policy import HybridEBLightGBMPolicy, LightGBMTLearnerPolicy
from solution.policy import MyPolicy

register("my_policy", MyPolicy)

# Worked example of using a FeaturePipeline inside a policy (see solution/feature_demo.py).
register("feature_demo", TimeOfDayPolicy)

register("best_fixed", BestFixedVariantPolicy)
register("seg_country", lambda seed=0: AdditiveEBPolicy(seed=seed, features=["country"], crosses=[]))
register("seg_eb", lambda seed=0: AdditiveEBPolicy(seed=seed))
register("seg_eb_recency", lambda seed=0: AdditiveEBPolicy(seed=seed, recency_halflife_days=4.0))
register("gbm_tlearner", GBMTLearnerPolicy)
register("hybrid_eb_gbm", HybridEBGBMPolicy)
register("lgbm_tlearner", LightGBMTLearnerPolicy)
register("hybrid_eb_lgbm", HybridEBLightGBMPolicy)

# Register every policy you want us to see — keep the variants you tried across the
# challenge, each under its own name (add a module per policy as you go), e.g.:
#   from solution.my_policy_v2 import MyPolicyV2
#   register("my_policy_v2", MyPolicyV2)
