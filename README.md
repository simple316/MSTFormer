
# MSTFormer: Motion Inspired Spatial-temporal Transformer for Accurate Vessel Trajectory Prediction
This is a method for vessel trajectory prediction. It mainly improves the trajectory prediction performance by data augmentation, dynamic-aware attention, and knowledge-inspired loss function.


## Requirements
```
geopy==2.2.0
ipdb==0.13.9
matplotlib==3.5.1
numpy==1.22.4
numpy_ext==0.9.8
pandas==1.4.3
scipy==1.8.1
```

## Data
A simple dataset is provided for testing the code, or you can generate your own by rewriting data_loader.py.This project provides some commented out code that we use to process the data for reference.
## Run
After configuring the environment, just run main_MSTFomer.py directly. Also, you can test different data by changing the parameters inside. The file where the logs are saved can be changed by changing the path in log.py.
## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@misc{qiang2023mstformer,
      title={MSTFormer: Motion Inspired Spatial-temporal Transformer with Dynamic-aware Attention for long-term Vessel Trajectory Prediction}, 
      author={Huimin Qiang and Zhiyuan Guo and Shiyuan Xie and Xiaodong Peng},
      year={2023},
      eprint={2303.11540},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## Contact
If you have any questions, feel free to contact Huimin Qiang through Email (qianghuimin21@mails.ucas.ac.cn) or Github issues. Pull requests are highly welcomed!
## Acknowledgments
Thanks to NOAA for providing the raw data (ttps://coast.noaa.gov/htdata/CMSP/AISDataHandler/2021/), and thanks to everyone for their interest in this work!
