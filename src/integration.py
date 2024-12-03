import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_heatmaps(heatmap1, heatmap2, threshold1=0.5, threshold2=0.5):
    """
    将 heatmap1 的形状信息移动到 heatmap2 的位置。

    Args:
        heatmap1 (numpy.ndarray): 热力图1，提供形状信息。
        heatmap2 (numpy.ndarray): 热力图2，提供位置信息。
        threshold1 (float): 对 heatmap1 的阈值，用于提取高亮区域。
        threshold2 (float): 对 heatmap2 的阈值，用于提取高亮区域。

    Returns:
        numpy.ndarray: 融合后的最终热力图。
    """
    # 确保热力图是单通道
    heatmap1 = heatmap1.astype(np.float32)
    heatmap2 = heatmap2.astype(np.float32)

    # Step 1: 提取高亮区域的二值掩码
    mask1 = (heatmap1 > threshold1).astype(np.uint8)
    mask2 = (heatmap2 > threshold2).astype(np.uint8)

    # Step 2: 计算高亮区域的质心
    moments1 = cv2.moments(mask1)
    moments2 = cv2.moments(mask2)

    if moments1["m00"] == 0 or moments2["m00"] == 0:
        raise ValueError("热力图中没有足够高亮的区域来计算质心。")

    centroid1 = (int(moments1["m10"] / moments1["m00"]), int(moments1["m01"] / moments1["m00"]))
    centroid2 = (int(moments2["m10"] / moments2["m00"]), int(moments2["m01"] / moments2["m00"]))

    # Step 3: 计算质心偏移量
    dx = centroid2[0] - centroid1[0]
    dy = centroid2[1] - centroid1[1]

    # Step 4: 对 mask1 进行平移
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_mask1 = cv2.warpAffine(mask1, translation_matrix, (heatmap1.shape[1], heatmap1.shape[0]))

    # Step 5: 融合生成最终热力图
    # 保留 heatmap1 的形状信息，但位置对齐到 heatmap2
    result_heatmap = np.where(shifted_mask1 > 0, heatmap1, 0)

    # 对结果进行归一化（可选）
    # result_heatmap = result_heatmap / result_heatmap.max()

    return result_heatmap

# 读取一张单通道的热力图为heatmap1，读取另一张单通道的热力图为heatmap2
heatmap1 = cv2.imread('heatmap1.png', cv2.IMREAD_GRAYSCALE)
heatmap2 = cv2.imread('heatmap2.png', cv2.IMREAD_GRAYSCALE)

# 调用函数
result = align_heatmaps(heatmap1, heatmap2, threshold1=0.6, threshold2=0.5)

# 可视化结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Heatmap1")
plt.imshow(heatmap1, cmap="hot")
plt.subplot(1, 3, 2)
plt.title("Heatmap2")
plt.imshow(heatmap2, cmap="hot")
plt.subplot(1, 3, 3)
plt.title("Result Heatmap")
plt.imshow(result, cmap="hot")
plt.show()
