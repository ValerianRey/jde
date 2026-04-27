echo "Running version $3 of the full pipeline for $1 on seed $2, subset faster_$2"
uv run iwrm_study $1 faster_$2 22 --lr-csv-path="default_lr_ranges_upgrad_mean.csv" --batch-size=64 --seed=$2 --wandb-mode=online --name="$1 $3 coarse $2"
sleep 60  # Just to be sure that wandb had time to receive the data
uv run download_study "$1 $3 coarse $2"
uv run refine_lr_range "$1 $3 coarse $2"
uv run iwrm_study $1 faster_$2 50 --lr-csv-path="$1 $3 coarse $2/refined_lr_ranges.csv" --batch-size=64 --seed=$2 --wandb-mode=online --name="$1 $3 fine $2"
sleep 60  # Just to be sure that wandb had time to receive the data
uv run download_study "$1 $3 fine $2"
echo "Done"
