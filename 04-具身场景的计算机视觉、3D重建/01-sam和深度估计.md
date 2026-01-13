
# 01-SAM 分割与单目深度估计：从二维感知到三维理解

在具身智能（Embodied AI）场景中，机器人仅“看清”物体（分割）是不够的，还需要知道物体“在哪里”（深度）。本章节将结合 Meta 的 **Segment Anything Model (SAM)** 和 **单目深度估计 (Monocular Depth Estimation)**，实现对场景的 3D 语义感知。

---

## 一、 为什么需要结合 SAM 与深度估计？

单纯的分割或单纯的深度图在机器人交互中都有局限性，两者的结合是实现 **3D Object Awareness（三维物体感知）** 的关键。

### 1.1 视觉与功能对比

| 模态 | 视觉效果 | 机器人能获取的信息 | 局限性 |
| :--- | :--- | :--- | :--- |
| **仅 SAM (分割)** | 清晰的物体轮廓 | "这里有一个杯子" (2D 区域) | **无距离感**。机器人不知道杯子是在桌边还是在房间另一头，无法规划抓取路径。 |
| **仅 Depth (深度)** | 模糊的颜色热力图 | "这一块区域离我比较近" | **无物体概念**。深度图边缘模糊，机器人难以区分"杯子"和"桌子"的边界，容易抓空。 |
| **SAM + Depth** | **轮廓 + 距离** | **"这个轮廓清晰的杯子，中心距离我 0.8米"** | **互补增强**。SAM 提供了精确的边缘（修正深度图的边缘模糊），深度图提供了空间位置。 |

### 1.2 评测标准 (Metrics)

在学术界和工程落地中，我们通常关注以下指标：

1.  **分割质量 (Segmentation Quality)**:
    * **mIoU (mean Intersection over Union)**: SAM (ViT-H) 通常能达到 **90%+** 的 IoU，意味着它扣出的物体边缘极度贴合真实物体。
2.  **深度准确性 (Depth Accuracy)**:
    * **RMSE (均方根误差)**: 衡量预测深度与真实深度的偏差。
    * **AbsRel (绝对相对误差)**: 越低越好。
    * *注意*: 单目深度估计通常输出**相对深度 (Relative Depth)**（如 0~1 的值），在实际应用中，通常需要通过相机参数或一个已知参考点将其对齐到真实世界尺度（米）。

---

## 二、 交互式实战：点击分割与测距 (Interactive Demo)

为了直观体验“指哪打哪”的效果，我们提供一个交互式脚本。
**功能**：加载图片 -> 鼠标点击物体 -> 实时生成 Mask -> 融合深度图计算该物体的平均距离。

### 2.1 脚本代码 (`interactive_sam_depth.py`)（请先完成第四点环境配置）

创建一个新文件 `interactive_sam_depth.py` 并运行：

```python
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import os

# --- 配置 ---
# 请确保下载了权重文件: sam_vit_h_4b8939.pth
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth" 
MODEL_TYPE = "vit_h"
DEPTH_MODEL_NAME = "Intel/dpt-large"
IMAGE_PATH = "test_image.jpg"  # 替换为你的本地图片路径

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# --- 1. 模型加载 ---
print("Loading Depth Model...")
depth_processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
depth_model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME).to(device)
depth_model.eval()

print("Loading SAM Model...")
if not os.path.exists(SAM_CHECKPOINT):
    raise FileNotFoundError(f"SAM checkpoint not found at {SAM_CHECKPOINT}. Please download it first.")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)

# --- 2. 核心处理函数 ---
def get_depth_map(image_pil):
    """生成全图的相对深度图"""
    inputs = depth_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # 插值到原图尺寸
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_pil.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    return prediction.squeeze().cpu().numpy()

# --- 3. 交互逻辑 ---
def interactive_demo():
    # 读取图像
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image {IMAGE_PATH} not found.")
        return

    image_cv = cv2.imread(IMAGE_PATH)
    image_cv = cv2.resize(image_cv, (1024, int(1024 * image_cv.shape[0] / image_cv.shape[1]))) # 调整大小以免屏幕放不下
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    # 预计算深度图
    print("Computing depth map...")
    depth_map = get_depth_map(image_pil)
    
    # 初始化 SAM
    print("Setting image for SAM...")
    predictor.set_image(image_rgb)

    # 状态变量
    input_point = []
    input_label = []
    current_mask = None
    
    print("\n--- 操作指南 ---")
    print("左键: 点击选择物体 (前景)")
    print("右键: 点击排除区域 (背景)")
    print("R键: 重置选择")
    print("Q键: 退出")

    def mouse_callback(event, x, y, flags, param):
        nonlocal input_point, input_label, current_mask
        
        if event == cv2.EVENT_LBUTTONDOWN:
            input_point.append([x, y])
            input_label.append(1) # 1 表示前景
        elif event == cv2.EVENT_RBUTTONDOWN:
            input_point.append([x, y])
            input_label.append(0) # 0 表示背景
        else:
            return

        # 调用 SAM 预测
        masks, scores, _ = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=False
        )
        current_mask = masks[0]
        
        # 计算深度统计
        # 提取 Mask 区域对应的深度值
        masked_depth = depth_map[current_mask]
        if masked_depth.size > 0:
            avg_depth = np.mean(masked_depth)
            print(f"Object Depth -> Mean: {avg_depth:.4f} (Score: {scores[0]:.3f})")

    cv2.namedWindow("SAM + Depth Interaction", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM + Depth Interaction", mouse_callback)

    while True:
        display_img = image_cv.copy()

        # 绘制点击点
        for pt, label in zip(input_point, input_label):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display_img, tuple(pt), 5, color, -1)

        # 绘制 Mask 叠加
        if current_mask is not None:
            # 创建红色半透明遮罩
            colored_mask = np.zeros_like(display_img)
            colored_mask[current_mask] = [0, 0, 255] # BGR format
            display_img = cv2.addWeighted(display_img, 1, colored_mask, 0.5, 0)
            
            # 绘制轮廓
            contours, _ = cv2.findContours(current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (255, 255, 255), 2)

        cv2.imshow("SAM + Depth Interaction", display_img)
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            input_point = []
            input_label = []
            current_mask = None
            print("Reset.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    interactive_demo()

```

---

## 三、 核心原理讲解

### 3.1 Segment Anything Model (SAM)

SAM 是计算机视觉领域的“瑞士军刀”。

* **图像编码器 (Image Encoder)**: 基于 ViT-H (Vision Transformer Huge)，负责对图像进行深度特征提取。这是最耗时的部分，但一张图只需要运行一次。
* **提示编码器 (Prompt Encoder)**: 将用户的点击（点）、框选（Box）或语言描述转化为向量。
* **掩码解码器 (Mask Decoder)**: 极其轻量级。结合图像特征和提示特征，**毫秒级**生成分割掩码。这就是为什么上面的交互式 Demo 可以做到实时响应。

### 3.2 单目深度估计 (Monocular Depth Estimation)
我们使用的是 DPT (Dense Prediction Transformer)。

* **原理**: 既然是单目（一个摄像头），模型无法像双目相机那样通过视差计算深度。DPT 实际上是学习了“上下文线索”：
    * **遮挡关系**: 物体A挡住物体B，A更近。
    * **透视关系**: 道路远端汇聚成一点。
    * **相对大小**: 同样是车，看着小的离得远。

> **⚠️ 重要提示：关于深度图颜色的反直觉现象**
> 你可能会发现生成的深度图中，**越近的物体颜色越亮（数值越大），越远的地方颜色越黑（数值越小）**。
> * 这是因为模型输出的是 **“逆深度” (Inverse Depth, $1/\text{distance}$)**。
> * **天空/远山**: 距离无穷远，$\frac{1}{\infty} \approx 0$，所以是 **黑色 (0)**。
> * **近处物体**: 距离很小，分母小导致数值大，所以是 **高亮黄色**。
> * **记忆口诀**: **“越亮越近，越黑越远”**。

![image-20260113211428842](./assets/image-20260113211428842.png)



---

## 四、 环境配置

要运行上述代码，需要安装 PyTorch, Transformers 以及 SAM 库。

1. **下载 SAM 模型权重 (ViT-H)**
```bash
# 文件大小约为 2.4 GB
wget 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
(https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

```


2. **安装 Python 依赖**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
(https://github.com/facebookresearch/segment-anything.git)
pip install transformers opencv-python matplotlib torch torchvision

```



---

## 五、 批处理流程代码 (自动处理脚本)

如果有一批图片需要自动生成 Mask 和 深度图（而非交互式点击），请使用以下脚本。
**优化说明**：已修复文件保存路径，确保“仅深度”、“仅SAM”、“两者结合”的结果分别保存，互不覆盖。

```python
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

# --- 模型导入 ---
from transformers import DPTImageProcessor, DPTForDepthEstimation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# --- 辅助函数：显示并保存结果 ---
def save_visualization(image, mask_or_depth, mode="sam", output_name="output.png"):
    plt.figure(figsize=(12, 8))
    
    if mode == "depth":
        # 并排显示：原图 vs 深度图
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask_or_depth, cmap="inferno")
        plt.colorbar(label="Relative Depth")
        plt.title("Depth Estimation")
        plt.axis('off')

    elif mode == "sam":
        # 叠加显示 SAM Mask
        plt.imshow(image)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        
        # 将 Mask 按面积排序，大的在下，小的在上
        sorted_anns = sorted(mask_or_depth, key=(lambda x: x['area']), reverse=True)
        
        img_overlay = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img_overlay[:,:,3] = 0 # 透明度初始化
        
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.4]]) # 随机颜色 + 0.4 透明度
            img_overlay[m] = color_mask
        ax.imshow(img_overlay)
        plt.title("SAM Segmentation")
        plt.axis('off')
    
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"结果已保存至: {output_name}")

# --- 主流程 ---
def main():
    # 1. 路径设置
    rgb_path = "image_d61af3.jpg" # 替换为图片
    sam_ckpt = "sam_vit_h_4b8939.pth"
    
    if not os.path.exists(rgb_path):
        print(f"错误: 找不到图片 {rgb_path}")
        return

    # 加载图片
    image_pil = Image.open(rgb_path).convert("RGB")
    image_np = np.array(image_pil)

    # ---------------------------------------------------------
    # 任务 1: 深度估计 (Depth Estimation)
    # ---------------------------------------------------------
    print("\n--- [1/2] 正在运行深度估计 ---")
    try:
        depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
        
        inputs = depth_processor(images=image_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
            
        # 插值还原尺寸
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
        # 保存深度图结果
        save_visualization(image_np, prediction, mode="depth", output_name="result_01_depth.png")
        
    except Exception as e:
        print(f"深度估计失败: {e}")

    # ---------------------------------------------------------
    # 任务 2: SAM 全图分割 (Segment Anything)
    # ---------------------------------------------------------
    print("\n--- [2/2] 正在运行 SAM 分割 ---")
    if os.path.exists(sam_ckpt):
        try:
            sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device)
            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(image_np)
            
            # 保存 SAM 结果 (注意文件名不同，避免覆盖)
            save_visualization(image_np, masks, mode="sam", output_name="result_02_sam_seg.png")
            
        except Exception as e:
            print(f"SAM 分割失败: {e}")
    else:
        print(f"跳过 SAM: 未找到权重文件 {sam_ckpt}")

if __name__ == "__main__":
    main()

```

