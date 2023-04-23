 # MSTT-STVSR (JAG 2022)
 ### 📖[**Paper**](https://www.sciencedirect.com/science/article/pii/S1566253523001100) | 🖼️[**PDF**](/figures/MSTT.pdf)

PyTorch codes for "[Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer](https://www.sciencedirect.com/science/article/pii/S0303243422000575)", **International Journal of Applied Earth Observation and Geoinformation (JAG)**, 2022.

[Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Jiang He](https://jianghe96.github.io/), [Qiang Zhang](https://qzhang95.github.io/), Jiang Sun, Xin Su, Jialian Wu, and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
Wuhan University  

### :tada::tada: News :tada::tada:
- Our MSTT is awarded as <font color=#FF000 >**ESI Highly Cited Paper**</font> (TOP 1%)!
# The overall network
 ![image](/figures/network.png)
 
 # Environment
 * CUDA 10.0
 * pytorch >=1.2
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 * build [PyFlow](https://github.com/pathak22/pyflow)
 
 # Dataset Preparation
 We reorganize the satellite video super-resolution data set named [*Jilin-189*](https://pan.baidu.com/s/1Y1-mS5gf7m8xSTJQPn4WZw) proposed in our previous work [MSDTGP](https://github.com/XY-boy/MSDTGP) to ensure the data directory structure is consistent with the [*Vimeo-90K*](http://toflow.csail.mit.edu/). 
 Finally, We obtained 2,647 video clips as a training set.
 
 Please download our dataset [*189_vime7*](https://pan.baidu.com/s/1Nx7lsS4371AVvrbkABSmmQ) from Baidu Netdisk. Code: 0rc2
 
 You can also train your dataset following the directory sturture below!
 
## Data directory structure
trainset--  
&emsp;|&ensp;189_vime7--  
&emsp;&emsp;|&ensp;train---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 006.png  
 
# Training
```
python main.py
```

# Test
```
python test.py
```
# Quantitive results
 ![image](/figures/result.png)
 
 # Qualitive results
 ![image](/figures/fig5.png)
*More details can be found in our paper!*

# Citation
If you find our work helpful, please consider to cite it, thank you very much!  
```
@article{xiao2022space,
  title={Space-time super-resolution for satellite video: A joint framework based on multi-scale spatial-temporal transformer},
  author={Xiao, Yi and Yuan, Qiangqiang and He, Jiang and Zhang, Qiang and Sun, Jing and Su, Xin and Wu, Jialian and Zhang, Liangpei},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={108},
  pages={102731},
  year={2022},
  publisher={Elsevier}
}
```

# Acknowledgement
Our work is built upon our previous work [MSDTGP](https://github.com/XY-boy/MSDTGP) and [STTN](https://github.com/researchmm/STTN).  
Thanks to the author for the source code !
