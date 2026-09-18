#!/bin/sh
# A stand-in for the operator's driver on a BUSY desktop (card t_8cb0a05e rehearsal).
#
# The card's host reported 8192 MiB total and only 1112 MiB free (the desktop held ~6.8 GB), which
# is the precondition the fix must survive. Answers every shape the production code asks for:
#
#   nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits   (one call)
#   nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits               (E1a budget)
#   nvidia-smi -L                                                                   (detection)
#   nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv        (gate facts)
TOTAL_MIB="${GGUFONE_FAKE_TOTAL_MIB:-8192}"
USED_MIB="${GGUFONE_FAKE_USED_MIB:-7080}"
FREE_MIB=$((TOTAL_MIB - USED_MIB))

ARGS="$*"
case "$ARGS" in
  *"memory.total,memory.used,memory.free"*) printf '%d MiB, %d MiB, %d MiB\n' "$TOTAL_MIB" "$USED_MIB" "$FREE_MIB" ;;
  *"memory.total,memory.free"*)             printf '%d, %d\n' "$TOTAL_MIB" "$FREE_MIB" ;;
  *"memory.total"*)                         printf '%d\n' "$TOTAL_MIB" ;;
  *"memory.free"*)                          printf '%d\n' "$FREE_MIB" ;;
  *"-L"*)                                   echo "GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-fake)" ;;
  *)                                        echo "NVIDIA-SMI 580.00  Driver Version: 580.00  CUDA Version: 13.0"
                                            echo "GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-fake)" ;;
esac
exit 0
