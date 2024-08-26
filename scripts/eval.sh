#! /bin/bash

cuda_device=0
# Array of folders
folders=(
  "./fid-data/example_inferred_folder"
)
parent_folder_name=$(date -u +%Y%m%d)
ref_dir="<path_to_reference_cropresized_mscoco2014_image_dir>"
# Iterate through each folder
for folder in "${folders[@]}"; do
    echo "Processing $folder"

    resized_folder=./fid-data256cropbutkeeprefpng/$parent_folder_name/$(basename "$folder")
    echo "Resizing $folder to $resized_folder"

    echo -e "\e[33mpython tools/resize.py main $folder $resized_folder\e[0m"
    python tools/resize.py main "$folder" "$resized_folder"

    echo -e "\e[33mCUDA_VISIBLE_DEVICES=$cuda_device python eval/fid.py $resized_folder --ref-dir=$ref_dir --no-crop\e[0m"
    CUDA_VISIBLE_DEVICES=$cuda_device python eval/fid.py "$resized_folder" --ref-dir="$ref_dir" --no-crop

    echo -e "\e[33mCUDA_VISIBLE_DEVICES=$cuda_device python eval/clip_score.py $resized_folder\e[0m"
    CUDA_VISIBLE_DEVICES=$cuda_device python eval/clip_score.py "$resized_folder" --use-dataloader --batch-size=1024

    echo -e "\e[33mCUDA_VISIBLE_DEVICES=$cuda_device fidelity --gpu 0 --isc --samples-find-deep --input1 $resized_folder\e[0m"
    echo "$folder" >> metrics/is30k.txt; CUDA_VISIBLE_DEVICES=$cuda_device fidelity --gpu 0 --isc --samples-find-deep --input1 "$resized_folder" >> metrics/is30k.txt

    #TODO: change env before running recall evaluation
    echo -e "\e[33mCUDA_VISIBLE_DEVICES=$cuda_device python eval/recall.py $ref_dir $resized_folder\e[0m"
    CUDA_VISIBLE_DEVICES=$cuda_device python eval/recall.py "$ref_dir" "$resized_folder"

    wait
done

echo "All processes completed."
