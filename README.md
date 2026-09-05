# Image-Conditioned Mask Diffusion with Hierarchical RWKV for
Medical Image Segmentation
Probabilistic Model
The codes for the work "Image-Conditioned Mask Diffusion with Hierarchical RWKV for
Medical Image Segmentation". I hope this will help you to reproduce the results.



## 2. Prepare data

### Melanoma Segmentation from Skin Images
1. Download ISIC dataset from https://challenge.isic-archive.com/data/. Your dataset folder under "data" should be like:

~~~
data
|   ----ISIC
|       ----Test
|       |   |   ISBI2016_ISIC_Part1_Test_GroundTruth.csv
|       |   |   
|       |   ----ISBI2016_ISIC_Part1_Test_Data
|       |   |       ISIC_0000003.jpg
|       |   |       .....
|       |   |
|       |   ----ISBI2016_ISIC_Part1_Test_GroundTruth
|       |           ISIC_0000003_Segmentation.png
|       |   |       .....
|       |           
|       ----Train
|           |   ISBI2016_ISIC_Part1_Training_GroundTruth.csv
|           |   
|           ----ISBI2016_ISIC_Part1_Training_Data
|           |       ISIC_0000000.jpg
|           |       .....
|           |       
|           ----ISBI2016_ISIC_Part1_Training_GroundTruth
|           |       ISIC_0000000_Segmentation.png
|           |       .....
~~~

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test


- Training

```bash
 
python scripts/segmentation_train.py --data_name ISIC --data_dir ./isic/ --out_dir ./tmp_out --image_size 224 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8 --lr_anneal_steps 50000 
```

- Sampling 

```bash
python scripts/segmentation_sample.py --data_name ISIC --data_dir ./isic/ --out_dir ./tmp_out --model_path ./tmp_out/emasavedmodel_0.9999_050000.pt --image_size 224 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5 
```

- Evaluation 

```bash
python scripts/segmentation_env.py --inp_pth ./tmp_out/ --out_pth ./isic/ISBI2016_ISIC_Part1_Test_GroundTruth/
```

## References and Acknowledgements
This code base uses certain code blocks and helper functions from:
* [EnsemDiff](https://github.com/JuliaWolleb/Diffusion-based-Segmentation)
* [MedSegDiff](https://github.com/KidsWithTokens/MedSegDiff)
* [MedSegDiff-V2](https://github.com/KidsWithTokens/MedSegDiff)
* [DiT](https://github.com/facebookresearch/DiT)
* [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)
* [Swin-Unet](https://github.com/Beckschen/TransUNet/tree/main) 



## Citation

Please cite
~~~

~~~
