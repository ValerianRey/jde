echo "Running version $2 of the full pipeline for svhn on seed $1, subset faster_$1"
pdm run iwrm_study svhn faster_$1 22 --optimizer=Adam --lr-csv-path="default_lr_ranges_upgrad_mean.csv" --epochs=20 --seed=$1 --wandb-mode=online --name="svhn $2 coarse $1"
sleep 60  # Just to be sure that wandb had time to receive the data
pdm run download_study "svhn $2 coarse $1"
pdm run refine_lr_range "svhn $2 coarse $1"
pdm run iwrm_study svhn faster_$1 50 --lr-csv-path="svhn $2 coarse $1/refined_lr_ranges.csv" --optimizer=Adam --epochs=20 --seed=$1 --wandb-mode=online --name="svhn $2 fine $1"
sleep 60  # Just to be sure that wandb had time to receive the data
pdm run download_study "svhn $2 fine $1"
echo "Done"
