 # MSTT-STVSR (JAG 2022)
 ### 📖[**Paper**](https://www.sciencedirect.com/science/article/pii/S0303243422000575) | 🖼️[**PDF**](/figures/MSTT.pdf)

PyTorch codes for "[Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer](https://www.sciencedirect.com/science/article/pii/S0303243422000575)", **International Journal of Applied Earth Observation and Geoinformation (JAG)**, 2022.

Authors: [Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Jiang He](https://jianghe96.github.io/), [Qiang Zhang](https://qzhang95.github.io/), Jing Sun, Xin Su, Jialian Wu, and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
Wuhan University  

### :tada::tada: News :tada::tada:
- Our MSTT is awarded as **ESI Highly Cited Paper (TOP 1%)!** [[Link](/figures/ESI.png)]
### Abstract
>Satellite video is an emerging type of earth observation tool, which has attracted increasing attention because of its application in dynamic analysis. However, most studies only focus on improving the spatial resolution of satellite video imagery. In contrast, few works are committed to enhancing the temporal resolution, and the joint spatial-temporal improvement is even less. The joint spatial-temporal enhancement can not only produce high-resolution imagery for subsequent applications, but also provide the potentials of clear motion dynamics for extreme events observation. In this paper, we propose a joint framework to enhance the spatial and temporal resolution of satellite video simultaneously. Firstly, to alleviate the problem of scale variation and scarce motion in satellite video, we design a feature interpolation module that deeply couples optical flow and multi-scale deformable convolution to predict unknown frames. Deformable convolution can adaptively learn the multi-scale motion information and profoundly complement optical flow information. Secondly, a multi-scale spatial-temporal transformer is proposed to aggregate the contextual information in long-time series video frames effectively. Since multi-scale patches are embedded in multiple heads for spatial-temporal self-attention calculation, we can comprehensively exploit multi-scale details in all frames. Extensive experiments on the Jilin-1 satellite video demonstrate that our model is superior to the existing methods. The source code is available at https://github.com/XY-boy.

### Overall
 ![image](/figures/network.png)
 
 ## Environment
 * CUDA 10.0
 * pytorch >=1.2
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 * build [PyFlow](https://github.com/pathak22/pyflow)
 
 ## Dataset Preparation
 We reorganize the satellite video super-resolution data set named [*Jilin-189*](https://pan.baidu.com/s/1Y1-mS5gf7m8xSTJQPn4WZw) proposed in our previous work [MSDTGP](https://github.com/XY-boy/MSDTGP) to ensure the data directory structure is consistent with the [*Vimeo-90K*](http://toflow.csail.mit.edu/). 
 Finally, We obtained 2,647 video clips as a training set.
 
 Please download our dataset [*189_vime7*](https://pan.baidu.com/s/1Nx7lsS4371AVvrbkABSmmQ) from Baidu Netdisk. Code: 0rc2
 
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
## Quantitative results
 ![image](/figures/result.png)
 
## Qualitative results
 ![image](/figures/fig5.png)
**More details can be found in our paper!**

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: xiao_yi@whu.edu.cn  
Tel: (+86) 15927574475 (WeChat)

## Citation
If you find our work helpful, please consider citing it. Thank you very much! (っ•̀ω•́)っ❥❥❥❥⁾⁾ 
```
@article{xiao2022mstt,
  title={Space-time super-resolution for satellite video: A joint framework based on multi-scale spatial-temporal transformer},
  author={Xiao, Yi and Yuan, Qiangqiang and He, Jiang and Zhang, Qiang and Sun, Jing and Su, Xin and Wu, Jialian and Zhang, Liangpei},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={108},
  pages={102731},
  year={2022},
  publisher={Elsevier}
}
```

## Acknowledgement
Our work is built upon our previous work [MSDTGP](https://github.com/XY-boy/MSDTGP) and [STTN](https://github.com/researchmm/STTN).  
Thanks to the author for the source code!
