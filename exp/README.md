# module
- edge_mask.py 这个脚本是测试在mask基础上提取边缘特征
- Gaussian.py 这个是参考LEGNet的LoG模块和高层的gaussian处理的实现
- tools是可视化工具

# net
- TGRSNet-prom3-10.py 对应原始(MSHNet)单流网络结构设计
- TGRSNet-prom3-8.py 对应最简单的双编码网络设计(add)

# module
- CDConv 
  - 中心差分卷积
- edge_mask
  - 从原图中提取的边缘特征,可以可视化并保存edge图片
- guassian
  - 高斯处理模块
- featurefusion
  - 双编码器中的edge branch和sptial branch的融合测试代码

# anlyze
- get_feature_npy
  - 这是一个用于获取各层网路特征的文件，输入模型和数据，输出各层对应的npy文件
  - 使用是要导入对应的模型代码
- simple_vis_npy、complex_vis_npy
  - 基于上上面的npy文件可以是实现简单和复杂的特征可视化
- paper_style_vis
  - 这是freqfusion论文风格底层特征可视化