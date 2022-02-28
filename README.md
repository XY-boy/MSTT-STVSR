# MSTT-STVSR
# Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer (JAG)
## Introuction
This is the official implementation of our paper [Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer](https://www.sciencedirect.com/science/article/pii/S0303243422000575) (MSTT_STVSR) published on ***International Journal of Applied Earth Observation and Geoinformation*** ([**JAG**](https://www.journals.elsevier.com/international-journal-of-applied-earth-observation-and-geoinformation))
### The network structure  
 ![image](/figures/network.png)
 
 ### Quantitive results
 ![image](/figures/fig5.png)
 
 ### Qualitive results
 ![image](/figures/result.png)
 ### More details can be found in our paper! [Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer](https://www.journals.elsevier.com/international-journal-of-applied-earth-observation-and-geoinformation)
 ## Environment
 * CUDA 10.0
 * pytorch >=1.2
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 * build [PyFlow](https://github.com/pathak22/pyflow)
 
 ## Dataset Preparation
 We reorganize the satellite video super-resolution data set named [*Jilin-189*](https://pan.baidu.com/s/1Y1-mS5gf7m8xSTJQPn4WZw) proposed in our previous work [MSDTGP](https://github.com/XY-boy/MSDTGP) to ensure the data directory structure is consistent with the [*Vimeo-90K*](http://toflow.csail.mit.edu/). 
 Finally, We obtained 2,647 video clips as a training set.
 
 Please download our dataset [*189_vime7*](https://pan.baidu.com/s/1Nx7lsS4371AVvrbkABSmmQ) from Baidu Netdisk. Code:0rc2
 You can also train your dataset following the directory sturture below!
 
### Data directory structure
trainset--  
&emsp;|&ensp;189_vime7--  
&emsp;&emsp;|&ensp;train---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 006.png  
 
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
@article{XIAO2022102731,
title = {Space-time super-resolution for satellite video: A joint framework based on multi-scale spatial-temporal transformer},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {108},
pages = {102731},
year = {2022},
issn = {0303-2434},
doi = {https://doi.org/10.1016/j.jag.2022.102731},
url = {https://www.sciencedirect.com/science/article/pii/S0303243422000575},
}
```

## Acknowledgement
Our work is built upon our previous work [MSDTGP](https://github.com/XY-boy/MSDTGP) and [STTN](https://github.com/researchmm/STTN).  
Thanks to the author for the source code !



 


