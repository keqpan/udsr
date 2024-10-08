# Unpaired Depth Super-Resolution in the Wild
This is the official implementation in `Python` and `PyTorch` of the [**Unpaired Depth Super-Resolution in the Wild**](https://ieeexplore.ieee.org/document/10637328) paper.

### Dataset preparation
First of all you need to download ScanNet dataset and InteriorNet dataset (if needed).

Notebook for exporting depth, image, poses and intrinsics from ScanNet's .sens data and following rendering is located at 
```/scannet_rendering/render_scannet.ipynb```

Notebook for filtering ScanNet and creating clear crops is located at
```/scannet_rendering/filtering.ipynb```

Lists for filenames for Scannet train/test split are located in ```/split_file_names``` folder.

### Folder structure and options
You shouldn't have any special structure for your data, but in train and test running commands you have to add ``` --path_A --path_B --path_A_test --path_B_test --A_add_paths (for train A images) --B_add_paths (for train B images) --A_add_paths_test (for test A images) --B_add_paths_test (for test B images)``` , or you can set this paths as default paths in ```options/train_options.py```; it can be more convenient. 

```--path_to_intr``` - is a folder where you exported ScanNet's depth, image, poses and intrinsics. It is the same folder as `output_path` from `render_scannet.ipynb`.


### Train Image Guidance Network 
To quick Image Guidance Network run: 
```sh
python main.py --name folder_for_saving_weights_name --gpu_ids 0,1,2,3 --display_freq 20 --print_freq 20 --n_epochs 150 --n_epochs_decay 150 --image_and_depth --continue_train --batch_size 12 --custom_pathes --w_real_l1 1 --w_syn_l1 1 --lr 0.0002 --Imagef_outf 128 --Imagef_basef 32 --use_scannet  --model I2D --norm_loss --do_train
```

##### Followed options are valid for Train Enhancement Component and Fine-tune Enhancement Network for super-resolution 
You can add ```--use_wandb``` to use wandb - good and easy-to-use logging tool. Before using it, you have to create an account on https://wandb.ai/ and log in on you machine.

You can add ```--do_test``` to see test set result after each epoch, or you can swich ```--do_train``` to ```--do_test``` and set   ```--n_epochs 1 --n_epochs_decay 0 --save_all --save_image_folder name_of_folder``` if you want to save predicted test images in your  `name_of_folder`.


### Train Translation Network
Translation Network is trained together with the whole Translation Component. To start training Translation Component, prepare dataset with the following structure for train folders as depicted:

    your_dataset_folder        # Your dataset folder name
    ├── trainA               
    │   ├── img                # folder with .jpg RGB images
    │   └── depth              # folder with .png depth maps (stored as uint16)
    ├── trainB
    │   ├── img                 
    │   └── depth   
    ├── testA
    └── testB
And similarly in should have `testA`, `testB`, `valA`, `valB`. Each of those folders should have `depth` and `img` folders inside as `trainA` and `trainB`. Then to specify path to your data, add `--dataroot path_to_your_dataset` to the command.


Then, run this command
```sh
python main.py --gpu_ids 0,1 --display_freq 20 --print_freq 20 --n_epochs 20 --n_epochs_decay 60 --custom_pathes  --use_scannet --lr 0.0002 --model translation_block --save_all --batch_size 6 --name translation --netD n_layers --crop_size_h 256 --crop_size_w 256 --do_train --dataroot path_to_your_dataset --max_distance 5100 --init_type xavier --model_type translation
```

### Train Enhancement Component
Before start training the whole Enhancement Component you have to end train Image Guidance Network and Translation Network. Then create a folder with location `checkpoints/your_folder_name` with the followed structure:

    .
    ├── ...
    ├── your_folder_name                  # Your folder name
    │   ├── latest_net_G_A_d.pth          # generator (HQ to LQ from translation). Copy corresponding checkpoint
    │   ├── latest_net_I2D_features.pth   # feature extractor from  Image Guidance Network
    │   └── latest_net_Image2Depth.pth    # main part (U-net) in Image Guidance Network
    └── ...

To start training process of Enhancement Component you have to run:
```sh
python main.py --gpu_ids 0,1,2,3 --display_freq 20 --print_freq 20 --n_epochs 20 --n_epochs_decay 60 --image_and_depth --continue_train --custom_pathes --use_image_for_trans --w_syn_l1 15 --w_real_l1_d 40  --norm_loss --w_syn_norm 2 --use_smooth_loss --w_smooth 1 --w_syn_holes 800 --w_real_holes 1600 --use_masked  --use_scannet --lr 0.0001 --model main_network_best --save_all --batch_size 6 --name your_folder_name --do_train --model_type main --use_wandb
```
After that you have to add flag `--no_aug` to turn off augmentation and fine-tune network on full-size RGBDs and continue training by the following command:

```sh
python main.py --gpu_ids 0,1,2,3 --display_freq 20 --print_freq 20 --n_epochs 10 --n_epochs_decay 20 --image_and_depth --continue_train --custom_pathes --use_image_for_trans --w_syn_l1 15 --w_real_l1_d 80  --norm_loss --w_syn_norm 3 --use_smooth_loss --w_smooth 1 --w_syn_holes 1600 --w_real_holes 1600 --use_masked  --use_scannet --lr 0.00002 --model main_network_best --save_all --batch_size 3 --name your_folder_name --model_type main --use_wandb --no_aug
```

If you use InteriorNet as HQ dataset you must use almost the same commands, you only have to change the weights as stated in article and additionally use `--interiornet` flag. 

### Fine-tune Enhancement Network for super-resolution 
Before training please copy your `your_folder_name` folder (to save enhancement results) and rename it (for example `your_sr_folder_name`)

To start fine-tuning process of the Enhancement Network for super-resolution you have to run:

```sh
python main.py --gpu_ids 0,1,2,3 --display_freq 20 --print_freq 20 --n_epochs 5 --n_epochs_decay 15 --image_and_depth --continue_train --custom_pathes --use_image_for_trans --w_syn_l1 15 --w_real_l1_d 80  --norm_loss --w_syn_norm 3 --use_smooth_loss --w_smooth 1 --w_syn_holes 1600 --w_real_holes 1600 --use_masked  --use_scannet --lr 0.00002 --model main_network_best --save_all --batch_size 1 --name your_sr_folder_name --do_train --crop_size_h 384 --crop_size_w 512 --use_wandb --model_type main --SR 
```
After that fine-tune a few epothes more with full-size images by setting `--crop_size_h 480 --crop_size_w 640` 

If you use InteriorNet as HQ dataset you must use almost the same commands, you only have to change weights as stated in the article and additionally use `--interiornet` flag. 


## Citation
```
@ARTICLE{safin2024udsr,
  author={Safin, Aleksandr and Kan, Maxim and Drobyshev, Nikita and Voynov, Oleg and Artemov, Alexey and Filippov, Alexander and Zorin, Denis and Burnaev, Evgeny},
  journal={IEEE Access}, 
  title={Unpaired Depth Super-Resolution in the Wild}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Superresolution;Training;Task analysis;Sensors;Noise measurement;Noise reduction;Image sensors;Depth data;enhancement;generative networks;super-resolution;unsupervised learning},
  doi={10.1109/ACCESS.2024.3444452}}
```




