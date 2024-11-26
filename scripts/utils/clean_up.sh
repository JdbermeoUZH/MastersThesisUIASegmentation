# Delete wandb cache
rm -r "${WANDB_CACHE_DIR}" &&
rm -r "${WANDB_DIR}"
echo "Wandb cache deleted"

if [ "$CLUSTER" = "bmic" ]; then
    echo "Nothing to do specific for bmic"
elif [ "$CLUSTER" = "euler" ]; then
    echo "Nothing to do specific for euler"
else
    echo "Unknown cluster"
fi