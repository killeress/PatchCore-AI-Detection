"""
dust_heatmap_metric 示意圖：Coverage vs IoU (4 場景)
基於 capi_inference.py 中 compute_dust_heatmap_iou / check_dust_per_region 的邏輯
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


def draw_scenario(ax, title, dust_center, dust_r, heat_center, heat_r,
                  metric, threshold=0.01, subtitle=""):
    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-2, 2.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Draw regions
    ax.add_patch(plt.Circle(dust_center, dust_r, color='#4A90D9', alpha=0.35))
    ax.add_patch(plt.Circle(heat_center, heat_r, color='#E74C3C', alpha=0.35))

    # Monte Carlo area estimation
    np.random.seed(42)
    n = 100000
    x = np.random.uniform(-1.5, 3.5, n)
    y = np.random.uniform(-2, 2.5, n)
    box = 5.0 * 4.5

    in_dust = ((x - dust_center[0])**2 + (y - dust_center[1])**2) <= dust_r**2
    in_heat = ((x - heat_center[0])**2 + (y - heat_center[1])**2) <= heat_r**2

    dust_area = np.sum(in_dust) / n * box
    heat_area = np.sum(in_heat) / n * box
    intersection = np.sum(in_dust & in_heat) / n * box
    union = np.sum(in_dust | in_heat) / n * box

    if metric == "coverage":
        value = intersection / dust_area if dust_area > 0 else 0.0
        formula = f"交集 / 灰塵面積\n= {intersection:.2f} / {dust_area:.2f}"
        metric_label = "Coverage"
    else:
        value = intersection / union if union > 0 else 0.0
        formula = f"交集 / (灰塵+熱區-交集)\n= {intersection:.2f} / {union:.2f}"
        metric_label = "IoU"

    is_dust = value >= threshold
    verdict = f">= {threshold} -> 判為灰塵 [DUST]" if is_dust else f"< {threshold} -> 保留為缺陷 [REAL]"
    color = '#27AE60' if is_dust else '#E74C3C'

    # Labels
    ax.text(dust_center[0], dust_center[1] + dust_r + 0.15, 'Dust', ha='center',
            fontsize=10, color='#2C3E50', fontweight='bold')
    ax.text(heat_center[0], heat_center[1] + heat_r + 0.15, 'Heatmap', ha='center',
            fontsize=10, color='#C0392B', fontweight='bold')

    # Result box
    result_text = f"{metric_label} = {value:.3f}\n{formula}\n{verdict}"
    props = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
    ax.text(1.0, -1.5, result_text, ha='center', va='center', fontsize=8.5,
            bbox=props, color='#2C3E50')

    if subtitle:
        ax.text(1.0, 2.2, subtitle, ha='center', fontsize=8.5, style='italic', color='#7F8C8D')

    ax.axis('off')


# ============================================================
# 2x4 layout: Row 1 = Coverage, Row 2 = IoU
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('dust_heatmap_metric 指標差異示意圖\n(基於 capi_inference.py 邏輯)',
             fontsize=16, fontweight='bold', y=0.98)

# --- Scenario params (shared between rows) ---
scenarios = [
    {
        "title_suffix": "熱區 >> 灰塵",
        "dust_center": (0.8, 0.3), "dust_r": 0.5,
        "heat_center": (1.2, 0.3), "heat_r": 1.2,
        "cov_sub": "灰塵被熱區蓋住 -> COV 高\n(不受熱區過大影響)",
        "iou_sub": "熱區太大 -> 聯集大 -> IoU 被稀釋\n!! 可能漏檢 (本應是灰塵)",
    },
    {
        "title_suffix": "熱區 << 灰塵",
        "dust_center": (1.0, 0.3), "dust_r": 1.2,
        "heat_center": (1.0, 0.3), "heat_r": 0.4,
        "cov_sub": "灰塵太大 -> 分母大 -> COV 低\n!! 小熱點被灰塵吃掉可能是誤判",
        "iou_sub": "聯集 ~ 灰塵面積 -> IoU 也低\n兩種模式結果相近",
    },
    {
        "title_suffix": "部分重疊",
        "dust_center": (0.3, 0.3), "dust_r": 0.8,
        "heat_center": (1.7, 0.3), "heat_r": 0.8,
        "cov_sub": "灰塵僅部分被蓋住 -> COV 中等",
        "iou_sub": "聯集包含兩者 -> IoU 偏低",
    },
    {
        "title_suffix": "不重疊",
        "dust_center": (-0.3, 0.3), "dust_r": 0.7,
        "heat_center": (2.3, 0.3), "heat_r": 0.7,
        "cov_sub": "熱區未蓋住灰塵 -> COV ~ 0\n-> 保留為真實缺陷",
        "iou_sub": "不重疊 -> IoU ~ 0\n-> 保留為真實缺陷",
    },
]

for i, s in enumerate(scenarios):
    # Row 1: Coverage
    draw_scenario(axes[0, i],
                  f"Coverage 場景 {i+1}：{s['title_suffix']}",
                  s["dust_center"], s["dust_r"],
                  s["heat_center"], s["heat_r"],
                  metric="coverage",
                  subtitle=s["cov_sub"])
    # Row 2: IoU
    draw_scenario(axes[1, i],
                  f"IoU 場景 {i+1}：{s['title_suffix']}",
                  s["dust_center"], s["dust_r"],
                  s["heat_center"], s["heat_r"],
                  metric="iou",
                  subtitle=s["iou_sub"])

# Row labels
fig.text(0.02, 0.72, 'Coverage\n模式', ha='center', va='center', fontsize=14,
         fontweight='bold', color='#2980B9',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8', edgecolor='#2980B9'))

fig.text(0.02, 0.30, 'IoU\n模式', ha='center', va='center', fontsize=14,
         fontweight='bold', color='#E74C3C',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#E74C3C'))

# Bottom legend
legend_text = (
    "【公式差異】\n"
    "  Coverage = 交集 / 灰塵面積             -> 只看「灰塵是否被熱區蓋住」，不管熱區多大\n"
    "  IoU      = 交集 / (灰塵+熱區-交集)     -> 標準重疊率，熱區越大分母越大，分數越低\n\n"
    "【關鍵場景差異】\n"
    "  場景1 (熱區>>灰塵): Coverage 正確判灰塵, IoU 被稀釋可能漏檢\n"
    "  場景2 (熱區<<灰塵): 兩者都低 -> 都會保留為缺陷 (行為一致)\n\n"
    "【代碼對應】capi_inference.py:\n"
    "  compute_dust_heatmap_iou()  -> 整體計算    |  check_dust_per_region() -> 逐連通區域判定\n"
    "  region_coverage >= iou_threshold -> 標記為灰塵，否則保留為真實缺陷"
)

fig.text(0.5, -0.02, legend_text, ha='center', va='top', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8F9FA', edgecolor='#BDC3C7', linewidth=1.5))

plt.tight_layout(rect=[0.05, 0.15, 1, 0.95])
output_path = r'C:\Users\rh.syu\Desktop\CAPI01_AD\dust_metric_diagram.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved to {output_path}")
