echo "Running version $2 of the full pipeline for cifar10 on seed $1, subset faster_$1"
pdm run iwrm_study cifar10 faster_$1 22 --optimizer=Adam --lr-csv-path="default_lr_ranges_upgrad_mean.csv" --epochs=15 --seed=$1 --wandb-mode=online --name="cifar10 $2 coarse $1"
sleep 60  # Just to be sure that wandb had time to receive the data
pdm run download_study "cifar10 $2 coarse $1"
pdm run refine_lr_range "cifar10 $2 coarse $1"
pdm run iwrm_study cifar10 faster_$1 50 --lr-csv-path="cifar10 $2 coarse $1/refined_lr_ranges.csv" --optimizer=Adam --epochs=15 --seed=$1 --wandb-mode=online --name="cifar10 $2 fine $1"
sleep 60  # Just to be sure that wandb had time to receive the data
pdm run download_study "cifar10 $2 fine $1"
echo "Done"
