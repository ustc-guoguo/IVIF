### 方法分类

<table border="1">
  <tr>
    <th>分类方式</th>
    <th>数据集</th>
    <th>说明</th>
  </tr>
  <tr>
    <td rowspan="4">任务角度</td>
    <td>面向视觉融合</td>
    <td>侧重低层次视觉质量（细节、对比度、纹理）</td>
  </tr>
  <tr>
    <td>任务驱动融合</td>
    <td>引入检测/分割/文本等高层任务或语义先验</td>
  </tr>
  <tr>
    <td>联合配准融合</td>
    <td>同时学习配准与融合，解决对齐问题</td>
  </tr>
  <tr>
    <td>退化鲁棒融合</td>
    <td>针对噪声、模糊、低照度等场景</td>
  </tr>

  <tr>
    <td rowspan="5">模型架构</td>
    <td>CNN</td>
    <td>...</td>
  </tr>
  <tr>
    <td>Vision Transformer</td>
    <td>...</td>
  </tr>
  <tr>
    <td rowspan="2">生成模型</td>
    <td>GAN</td>
  </tr>
  <tr>
    <td>Diffusion</td>
  </tr>
  <tr>
    <td>VLM</td>
    <td>...</td>
  </tr>
</table>


### 数据集

| Dataset | 地址 | 图片对数 | 分辨率 | color |
|---|---|---|---|---|
| TNO | https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029 | 261 | 768*576 | ❌ |
| RoadSence | https://github.com/hanna-xu/RoadScene | 221 | Various | ✅ |
| MSRS | https://github.com/Linfeng-Tang/MSRS | 2999 | 768*567 | ✅ |
| M3FD | https://github.com/JinyuanLiu-CV/TarDAL | 4200 | 1024*768 | ✅ |
| LLVIP | https://bupt-ai-cz.github.io/LLVIP/ | 16836 | 1280*720 | ✅ |


### 图像模式

> 大部分融合算法的处理流程：RGB -> YCbCr -> f=FusionNet(Y, if) -> fCbcr -> RGB
> 注：PIL默认打开方式RGB，而OpenCV默认打开方式为BGR，OpenCV：0 → 灰度模式 → (H, W)

| 类型 | 区别 | 最合适读取方式 |
|---|---|---|
| 灰度类（Grayscale / L） | 只有亮度（Y = 0.299R + 0.587G + 0.114B） | OpenCV |
| RGB类（直接存颜色） | R（红） + G（绿） + B（蓝） | PIL |
| 亮度-色度分离类（YCbCr / YUV） | Y（亮度）；Cb（蓝色差）；Cr（红色差）；人眼对亮度敏感，—对颜色不敏感 | RGB转换 |
| 离散整数（类别ID） | label | PIL |

### 评估指标

| 类型 | 地址 |
|---|---|
| Matlab代码1 | https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029 |
| Matlab代码2（全） | https://github.com/Linfeng-Tang/Evaluation-for-Image-Fusion/tree/main/Metric |
| Pytorch代码1 | https://github.com/RollingPlain/IVIF_ZOO/blob/main/Metric/Metric_torch.py |
| Python代码2（全） | https://gitee.com/WLLwssy/fusion_metric_python |
