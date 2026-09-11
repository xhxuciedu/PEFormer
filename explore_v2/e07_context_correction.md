# E07 - an explicit context-conditioned design correction, on held-out loci

187,102 candidate measurements in 51,200 decision groups over 23,431 locus groups and 16 contexts. Locus-grouped 5-fold; ridge penalty chosen inside each training split; fold 0 untouched.

## Held-out residual prediction

| model | Spearman | R^2 vs predicting zero |
|---|---:|---:|
| design_only | 0.3796 | +0.0550 |
| design_x_context | 0.4418 | +0.1023 |
| design_x_context_SHUFFLED | 0.4020 | +0.0599 |

Blend weight chosen on training loci by the decision metric, per fold: {'design_only': [0.1, 0.25, 0.1, 0.1, 0.25], 'design_x_context': [0.25, 0.25, 0.25, 0.25, 0.25], 'design_x_context_SHUFFLED': [0.1, 0.25, 0.1, 0.1, 0.25]}. A weight of zero recovers the frozen model exactly, so a useless correction can be switched off rather than forced on. The `_SELECTIVE` arms additionally choose, on training loci, what fraction of decision groups the correction may touch at all: {'design_only': [1.0, 1.0, 1.0, 1.0, 1.0], 'design_x_context': [1.0, 1.0, 1.0, 1.0, 1.0], 'design_x_context_SHUFFLED': [1.0, 1.0, 1.0, 1.0, 0.5]}.


## The decision

| score | groups | top-1 accuracy | selected efficiency | regret | mean Spearman (>=3) |
|---|---:|---:|---:|---:|---:|
| base | 43,425 | 0.6704 | 0.2921 | 0.01984 | 0.6953 |
| design_only | 43,425 | 0.6721 | 0.2924 | 0.01956 | 0.6936 |
| design_x_context | 43,425 | 0.6728 | 0.2925 | 0.01948 | 0.6939 |
| design_x_context_SHUFFLED | 43,425 | 0.6721 | 0.2924 | 0.01959 | 0.6929 |
| design_only_SELECTIVE | 43,425 | 0.6721 | 0.2924 | 0.01956 | 0.6936 |
| design_x_context_SELECTIVE | 43,425 | 0.6728 | 0.2925 | 0.01948 | 0.6939 |
| design_x_context_SHUFFLED_SELECTIVE | 43,425 | 0.6719 | 0.2924 | 0.01962 | 0.6935 |

Oracle within these groups is 0.3120 and a random pick gets 0.2131, so the whole decision is worth 0.0989 efficiency.

## Paired against the frozen model, resampling locus groups

| correction | groups | delta accuracy | 95% CI | delta regret | 95% CI |
|---|---:|---:|---|---:|---|
| design_only | 43,425 | +0.0018 | [+0.0006, +0.0028] | -0.00028 | [-0.00040, -0.00014] |
| design_x_context | 43,425 | +0.0024 | [+0.0009, +0.0038] | -0.00036 | [-0.00050, -0.00019] |
| design_x_context_SHUFFLED | 43,425 | +0.0018 | [+0.0005, +0.0028] | -0.00025 | [-0.00039, -0.00013] |
| design_only_SELECTIVE | 43,425 | +0.0018 | [+0.0006, +0.0028] | -0.00028 | [-0.00040, -0.00014] |
| design_x_context_SELECTIVE | 43,425 | +0.0024 | [+0.0009, +0.0038] | -0.00036 | [-0.00050, -0.00019] |
| design_x_context_SHUFFLED_SELECTIVE | 43,425 | +0.0016 | [+0.0004, +0.0026] | -0.00022 | [-0.00034, -0.00011] |

## By context

| context | groups | base acc | design x context acc | selective acc | shuffled selective acc |
|---|---:|---:|---:|---:|---:|
| pridict_pridict2|HEK293T|PE2 | 20,403 | 0.6049 | 0.6090 | 0.6090 | 0.6079 |
| pridict_pridict2|K562|PE2 | 6,523 | 0.7520 | 0.7509 | 0.7509 | 0.7515 |
| pridict_pridict2|K562|PE4 | 6,198 | 0.7854 | 0.7872 | 0.7872 | 0.7865 |
| deepprime|HEK293T|PE2 | 2,875 | 0.7252 | 0.7231 | 0.7231 | 0.7231 |
| deepprime|A549|PE2 | 1,335 | 0.6794 | 0.6824 | 0.6824 | 0.6824 |
| deepprime|A549|PE4 | 1,281 | 0.6136 | 0.6120 | 0.6120 | 0.6144 |
| deepprime|HEK293T|PE4 | 1,391 | 0.6147 | 0.6139 | 0.6139 | 0.6161 |
| pridict_pridict2|U2OS|PE2 | 256 | 0.4141 | 0.4258 | 0.4258 | 0.4141 |
| pridict_pridict2|U2OS|PE4 | 256 | 0.4141 | 0.4102 | 0.4102 | 0.4180 |
| deepprime|DLD1|PE4 | 749 | 0.7623 | 0.7664 | 0.7664 | 0.7610 |
| pridict_pridict2|HEK293T|PE4 | 130 | 0.4692 | 0.4846 | 0.4846 | 0.4769 |
| deepprime|DLD1|PE2 | 477 | 0.7757 | 0.7987 | 0.7987 | 0.7778 |
| deepprime|HeLa|PE2 | 427 | 0.7681 | 0.7728 | 0.7728 | 0.7728 |
| deepprime|HCT116|PE2 | 405 | 0.7679 | 0.7728 | 0.7728 | 0.7728 |
| deepprime|NIH3T3|PE4 | 384 | 0.7083 | 0.7109 | 0.7109 | 0.7057 |
| deepprime|MDA-MB-231|PE2 | 335 | 0.7075 | 0.7045 | 0.7045 | 0.7015 |