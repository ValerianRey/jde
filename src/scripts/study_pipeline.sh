echo "Running version $3 of the full pipeline for $1 on seed $2, subset faster_$2"
pdm run iwrm_study $1 faster_$2 22 --seed=$2 --wandb-mode=online --name="$1 $3 coarse $2"
sleep 60  # Just to be sure that wandb had time to receive the data
pdm run download_study "$1 $3 coarse $2"
pdm run refine_lr_range "$1 $3 coarse $2"
pdm run iwrm_study $1 faster_$2 50 --lr-csv-path="$1 $3 coarse $2/refined_lr_ranges.csv" --seed=$2 --wandb-mode=online --name="$1 $3 fine $2"
sleep 60  # Just to be sure that wandb had time to receive the data
pdm run download_study "$1 $3 fine $2"
echo "Done"
