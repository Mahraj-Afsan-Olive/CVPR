@echo off
set CUDA_VISIBLE_DEVICES=0
python eval.py ^
--dataset_name "G:\SAM2-UNet-main\Dataset" ^
--pred_path "G:\SAM2-UNet-main\predictions" ^
--gt_path "G:\SAM2-UNet-main\Dataset\test\mask"
