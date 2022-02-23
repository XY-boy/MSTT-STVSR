# MSTT-STVSR
# Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer (JAG)
## Introuction
This is the official implementation of our paper [Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer](https://www.journals.elsevier.com/international-journal-of-applied-earth-observation-and-geoinformation) (MSTT_STVSR) published on ***International Journal of Applied Earth Observation and Geoinformation*** (**JAG**). <font color="#FF0000">[Journal link](https://www.journals.elsevier.com/international-journal-of-applied-earth-observation-and-geoinformation)</font> 

### The network structure  
 ![image](/figures/network.png)
 
 ### Quantitive results
 ![image](/figures/fig5.png)
 
 ### Qualitive results
 ![image](/figures/result.png)
 #### More details can be found in our paper!
 ## Environment
 * CUDA 10.0
 * pytorch 1.x
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 * build [PyFlow](https://github.com/pathak22/pyflow)
 
 ## Dataset Preparation
 We reorganize the satellite video super-resolution data set named [Jilin-189](https://pan.baidu.com/s/1Y1-mS5gf7m8xSTJQPn4WZw) proposed in our previous work [MSDTGP](https://github.com/XY-boy/MSDTGP) to ensure the data directory structure is consistent with the Vimeo-90K.
 Please download our dataset! Code:31ct  
 You can also train your dataset following the directory sturture below!
 
### Data directory structure
trainset--  
&emsp;|&ensp;train--  
&emsp;&emsp;|&ensp;LR4x---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 099.png  
&emsp;&emsp;|&ensp;GT---   
&emsp;&emsp;|&ensp;Bicubic4x--- 

testset--  
&emsp;|&ensp;eval--  
&emsp;&emsp;|&ensp;LR4x---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 099.png  
&emsp;&emsp;|&ensp;GT---   
&emsp;&emsp;|&ensp;Bicubic4x--- 
 
 ## Training
```
python main.py
```

## Test
```
python test.py
```

## Citation
If you find our work helpful, please cite:  
```
@ARTICLE{9530280,  
author={Xiao, Yi and Su, Xin and Yuan, Qiangqiang and Liu, Denghong and Shen, Huanfeng and Zhang, Liangpei},  
journal={IEEE Transactions on Geoscience and Remote Sensing},  
title={Satellite Video Super-Resolution via Multiscale Deformable Convolution Alignment and Temporal Grouping Projection},   
year={2021},  
volume={},  
number={},  
pages={1-19},  
doi={10.1109/TGRS.2021.3107352}}
```

## Acknowledgement
Our work is built upon our previous work [MSDTGP](https://github.com/XY-boy/MSDTGP) and [STTN](https://github.com/researchmm/STTN).  
Thanks to the author for the source code !



 


