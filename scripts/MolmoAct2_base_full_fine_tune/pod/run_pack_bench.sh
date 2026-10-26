#!/usr/bin/env bash
# Packing throughput benchmark for the base-checkpoint MolmoAct2 full fine-tune.
cd /root/molmoact2-so101
source /root/molmoact2-venv/bin/activate
PACK='--dynamic_seq_len=false --seq_len=2100'
python bench.py \
  "name=base_full_bs16 FT_MODE=full FLOW_K=1 DEVICE_BS=16" \
  "name=base_pack2100_bs4 FT_MODE=full FLOW_K=1 DEVICE_BS=4 PACKING=true EXTRA_ARGS='$PACK'" \
  "name=base_pack2100_bs8 FT_MODE=full FLOW_K=1 DEVICE_BS=8 PACKING=true EXTRA_ARGS='$PACK'"
echo BENCH_EXIT=$?
