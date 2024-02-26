# SliceFormer: Deep Dense Depth Estimation from a Single Indoor Omnidirectional Image using a Slice-based Transformer


## Environment
Configure the environment according to the [environment configuration file](/config/SliceFormer_environment.txt).

## Data
Data can be downloaded from 3D60 official website or [here](https://drive.google.com/drive/folders/11eIOna43uLOOu91E3dDo6jt16LpP7LZE?usp=sharing)

## Model
Pre-trained models can be downloaded from [here](https://drive.google.com/file/d/1l5nOamm-Wgbec_EkrgHKuvcbNSKp3EgC/view?usp=sharing)

## Tools
You can find the [code](/tools/generate_random_normal_perspective.py) to generate the general perspective dataset from the datasets.

## Implementation

Decompress the data and place it in '/data'

1. Configure the environment following the instructions in `SliceFormer_environment.txt` located in the `/config` directory.

2. Once you have set up the environment and installed the necessary dependencies, proceed to configure the `project_path` variable in the `tools/infer.py` file.

### Run

Run the following command:

```bash
python tools/infer.py
```


### Own data

If you intend to use your own data, make sure to replace both the /data and /dataList directories with your own data.



## Reference
```
https://github.com/sunset1995/py360convert/tree/master/py360convert

@article{alhashim2018high,
  title={High quality monocular depth estimation via transfer learning},
  author={Alhashim, Ibraheem and Wonka, Peter},
  journal={arXiv preprint arXiv:1812.11941},
  year={2018}
}

@inproceedings{bhat2021adabins,
  title={Adabins: Depth estimation using adaptive bins},
  author={Bhat, Shariq Farooq and Alhashim, Ibraheem and Wonka, Peter},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4009--4018},
  year={2021}
}
```