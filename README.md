# DiffRWKV
The codes for the work "Medical Image Segmentation with Diffusion Probabilistic Model". I hope this will help you to reproduce the results.



## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (Synapse: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd and ACDC: https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4).

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 4. Use larger batch size if you have enough memory

- Train

```bash
 
python scripts/segmentation_train.py --data_name ISIC --data_dir ./isic/ --out_dir ./tmp_out --image_size 224 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8 --lr_anneal_steps 50000 
```

- Sample 

```bash
python scripts/segmentation_sample.py --data_name ISIC --data_dir ./isic/ --out_dir ./tmp_out --model_path ./tmp_out/emasavedmodel_0.9999_050000.pt --image_size 224 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5 
```

- Test 

```bash
python scripts/segmentation_env.py --inp_pth ./tmp_out/ --out_pth ./isic/ISBI2016_ISIC_Part1_Test_GroundTruth/
```

## References and Acknowledgements
This code base uses certain code blocks and helper functions from:
* [Diffusion Models for Implicit Image Segmentation Ensembles](https://github.com/JuliaWolleb/Diffusion-based-Segmentation)
* [MedSegDiff](https://github.com/KidsWithTokens/MedSegDiff)
* [MedSegDiff-V2](https://github.com/KidsWithTokens/MedSegDiff)
* [Swin-Unet](https://github.com/Beckschen/TransUNet/tree/main) 
* [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)


## Citation


