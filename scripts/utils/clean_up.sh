# Delete wandb cache
rm -r "${WANDB_CACHE_DIR}" &&
rm -r "${WANDB_DIR}"
echo "Wandb cache deleted"

if [ "$CLUSTER" = "bmic" ]; then
    
elif [ "$CLUSTER" = "euler" ]; then
    