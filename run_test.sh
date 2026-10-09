#!/bin/bash

#----------------------------------------------------------------
# TEST FOR COMPARISON
#----------------------------------------------------------------

# --- (mpe-pp) ---

# poam
# python src/main.py --config=open/open_train_pp \
# --alg-config=mpe/poam \
# --env-config=mpe \
# with t_max=3000000 \
# env_args.key="mpe:PredatorPrey-v0" \
# env_args.time_limit=200 \
# env_args.pretrained_wrapper="PretrainedTag" \
# ed_pretrain_ts=500000 \
# --seed=112358

# mat_naht
# python src/main.py --config=open/mat_naht/open_train_pp_mat_naht \
# --alg-config=mpe/mat_naht \
# --env-config=mpe \
# with t_max=3000000 \
# env_args.key="mpe:PredatorPrey-v0" \
# env_args.time_limit=200 \
# env_args.pretrained_wrapper="PretrainedTag" \
# ed_pretrain_ts=0 \
# --seed=112358

# --- (5v6) ---

# poam
# python src/main.py --config=open/open_train_5v6 \
# --alg-config=sc2/mat_naht \
# --env-config=sc2 \
# with t_max=3000000 \
# env_args.map_name=5m_vs_6m \
# ed_pretrain_ts=500000 \
# --seed=112358

# mat_naht
# python src/main.py --config=open/open_train_5v6 \
# --alg-config=sc2/mat_naht \
# --env-config=sc2 \
# with t_max=3000000 \
# env_args.map_name=5m_vs_6m \
# ed_pretrain_ts=500000 \
# --seed=112358

# --- (8v9) ---

# poam
# python src/main.py --config=open/open_train_8v9 \
# --alg-config=sc2/mat_naht \
# --env-config=sc2 \
# with t_max=3000000 \
# env_args.map_name=8m_vs_9m \
# ed_pretrain_ts=500000 \
# --seed=112358

# mat_naht
# python src/main.py --config=open/open_train_8v9 \
# --alg-config=sc2/mat_naht \
# --env-config=sc2 \
# with t_max=3000000 \
# env_args.map_name=8m_vs_9m \
# ed_pretrain_ts=500000 \
# --seed=112358

# --- (10v11) ---

# poam
# python src/main.py --config=open/open_train_10v11 \
# --alg-config=sc2/mat_naht \
# --env-config=sc2 \
# with t_max=3000000 \
# env_args.map_name=10m_vs_11m \
# ed_pretrain_ts=500000 \
# --seed=112358

# mat_naht
# python src/main.py --config=open/open_train_10v11 \
# --alg-config=sc2/mat_naht \
# --env-config=sc2 \
# with t_max=3000000 \
# env_args.map_name=10m_vs_11m \
# ed_pretrain_ts=500000 \
# --seed=112358

# --- (3sv5z) ---

# poam
# python src/main.py --config=open/open_train_3sv5z \
# --alg-config=sc2/mat_naht \
# --env-config=sc2 \
# with t_max=3000000 \
# env_args.map_name=3s_vs_5z \
# ed_pretrain_ts=500000 \
# --seed=112358

# mat_naht
# python src/main.py --config=open/open_train_3sv5z \
# --alg-config=sc2/mat_naht \
# --env-config=sc2 \
# with t_max=3000000 \
# env_args.map_name=3s_vs_5z \
# ed_pretrain_ts=500000 \
# --seed=112358

#----------------------------------------------------------------
#----------------------------------------------------------------