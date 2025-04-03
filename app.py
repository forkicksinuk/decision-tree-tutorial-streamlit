# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import accuracy_score # We might use this later
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# --- Page Configuration ---
st.set_page_config(
    page_title="决策树探秘之旅",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 决策树探秘之旅")
st.caption("一步步理解决策树如何进行分类")

# --- Helper Function for Plotting ---
def plot_data(X, y, split_feature=None, split_value=None, ax=None, title="数据分布"):
    """绘制二维数据点及可选的分割线"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    colors = {0: 'red', 1: 'blue'}
    markers = {0: 'o', 1: 's'}
    labels = {0: '类别 0', 1: '类别 1'}

    for class_value in np.unique(y):
        subset = X[y == class_value]
        ax.scatter(subset[:, 0], subset[:, 1],
                   c=colors[class_value],
                   label=labels[class_value],
                   marker=markers[class_value],
                   edgecolor='k', s=50)

    ax.set_xlabel("特征 X1")
    ax.set_ylabel("特征 X2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 绘制分割线
    if split_feature is not None and split_value is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if split_feature == 0: # Split on X1 (vertical line)
            ax.vlines(split_value, ymin=ylim[0], ymax=ylim[1], color='green', lw=3, linestyle='--')
            ax.text(split_value + 0.05 * (xlim[1]-xlim[0]), ylim[0] + 0.9 * (ylim[1]-ylim[0]),
                    f'X1 = {split_value:.2f}', color='green', ha='left')
        elif split_feature == 1: # Split on X2 (horizontal line)
            ax.hlines(split_value, xmin=xlim[0], xmax=xlim[1], color='green', lw=3, linestyle='--')
            ax.text(xlim[0] + 0.05 * (xlim[1]-xlim[0]), split_value + 0.05 * (ylim[1]-ylim[0]),
                    f'X2 = {split_value:.2f}', color='green', va='bottom')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return fig, ax

# --- Stage 1: 分类的直觉 ---
st.header("阶段 1: 分类的直觉 - 用规则区分")
st.markdown("""
想象一下我们有一些数据点，每个点属于两个类别（红色圆圈🔵 或 蓝色方块🟥）中的一个。
我们的目标是找到一些简单的“规则”来区分它们。

**任务:** 尝试调整下面的滑块，画出一条**垂直**或**水平**的分割线，看看是否能很好地把红点和蓝点分开。
""")

# 1.1 生成简单的二维数据
np.random.seed(42) # for reproducibility
X_simple = np.random.rand(50, 2) * 5
# 简单的线性规则: 如果 X1 > 2.5，则为类别 1 (蓝色)，否则为类别 0 (红色)
y_simple = (X_simple[:, 0] > 2.5).astype(int)
# 加入一些噪音
noise_indices = np.random.choice(len(X_simple), size=5, replace=False)
y_simple[noise_indices] = 1 - y_simple[noise_indices]


# 1.2 互动控件
col1_1, col1_2 = st.columns([1, 2])

with col1_1:
    st.subheader("选择分割规则")
    feature_map = {"特征 X1 (画垂直线)": 0, "特征 X2 (画水平线)": 1}
    selected_feature_name = st.radio("选择要依据的特征:", list(feature_map.keys()))
    selected_feature_idx = feature_map[selected_feature_name]

    # 根据所选特征设置滑块范围
    min_val = X_simple[:, selected_feature_idx].min()
    max_val = X_simple[:, selected_feature_idx].max()
    step = (max_val - min_val) / 50
    default_val = (min_val + max_val) / 2

    split_value = st.slider(f"设置 '{selected_feature_name.split(' ')[1]}' 的分割阈值:",
                            min_value=min_val, max_value=max_val, value=default_val, step=step)

with col1_2:
    st.subheader("数据和你的分割线")
    fig1, ax1 = plot_data(X_simple, y_simple,
                          split_feature=selected_feature_idx,
                          split_value=split_value,
                          title="简单数据集与你的分割尝试")
    st.pyplot(fig1)

st.markdown("""
**思考:**
*   你画的这条线能完美分开两种颜色的点吗？
*   只用一条线够吗？如果不够，可能需要怎么做？
*   选择哪个特征（X1 或 X2）和哪个阈值似乎分得更好？
""")
st.markdown("---")


# --- Stage 2: 决策树的样子 ---
st.header("阶段 2: 决策树的样子 - 像流程图一样思考")
st.markdown("""
决策树就像一个流程图，它把我们在阶段 1 中尝试的“规则”（提问）串联起来。

下面是一个针对上面简单数据集构建的**示例决策树**:
""")

# 2.1 预设一个简单的决策树 (DOT 语言)
# 这个树对应 X1 <= 2.5 的分割规则
dot_simple_tree = graphviz.Digraph(comment='简单决策树示例')
dot_simple_tree.node('0', 'X1 <= 2.61 ?\n(根节点)') # 实际分割点可能略有不同，这里用一个接近的值
dot_simple_tree.node('1', '预测: 红色 🔵\n(叶节点)')
dot_simple_tree.node('2', '预测: 蓝色 🟥\n(叶节点)')
dot_simple_tree.edge('0', '1', label='是 (True)')
dot_simple_tree.edge('0', '2', label='否 (False)')

st.graphviz_chart(dot_simple_tree)

st.markdown("""
**解读:**
1.  从 **根节点** 开始提问：“特征 X1 是否小于等于 2.61？”
2.  如果答案是 **是 (True)**，则沿着标有“是”的 **分支** 向左走，到达 **叶节点**，预测该点为 **红色 🔵**。
3.  如果答案是 **否 (False)**，则沿着标有“否”的 **分支** 向右走，到达另一个 **叶节点**，预测该点为 **蓝色 🟥**。

**互动演示:** 输入一个新数据点的坐标，看看它会沿着树的哪个路径走。
""")

col2_1, col2_2 = st.columns(2)

with col2_1:
    st.subheader("输入新数据点坐标")
    new_x1 = st.number_input("输入 特征 X1 的值:", value=1.5, step=0.1)
    new_x2 = st.number_input("输入 特征 X2 的值:", value=3.0, step=0.1)

with col2_2:
    st.subheader("决策路径分析")
    # 根据简单树的规则判断
    if new_x1 <= 2.61:
        st.success(f"1. **问题:** X1 ({new_x1:.2f}) <= 2.61 ?  **回答: 是 (True)**")
        st.info("   -> 沿着 '是' 分支走...")
        st.markdown("2. **到达叶节点:** 预测为 **红色 🔵**")
        final_prediction = "红色 🔵"
    else:
        st.success(f"1. **问题:** X1 ({new_x1:.2f}) <= 2.61 ?  **回答: 否 (False)**")
        st.info("   -> 沿着 '否' 分支走...")
        st.markdown("2. **到达叶节点:** 预测为 **蓝色 🟥**")
        final_prediction = "蓝色 🟥"

    # 可视化这个新点
    fig2, ax2 = plot_data(X_simple, y_simple, title="数据点与新输入的点")
    ax2.scatter(new_x1, new_x2, c='lime', marker='*', s=200, edgecolor='black', label=f'新点 ({new_x1:.1f}, {new_x2:.1f})\n预测: {final_prediction}')
    ax2.legend()
    st.pyplot(fig2)


st.markdown("""
**小结:** 决策树提供了一种结构化的方式来应用一系列规则，对数据进行分类。每个内部节点代表一个问题（基于某个特征的测试），每个分支代表一个答案，每个叶节点代表一个最终的分类预测。
""")
st.markdown("---")

# --- 后续阶段占位符 ---
# --- Stage 3: 决策的核心 - 如何选择“最好的”问题？ ---
st.header("阶段 3: 决策的核心 - 如何选择“最好的”问题？")
st.markdown("""
在阶段 1，我们凭直觉尝试分割数据。但机器如何**自动**找到“最好”的分割线呢？
决策树通过衡量数据的“**纯度**”或“**不纯度**”来做到这一点。一个好的分割应该让分割后的两个区域都尽可能“纯”（即包含的类别尽量单一）。

我们使用 **基尼不纯度 (Gini Impurity)** 来衡量这种混乱程度：
*   公式: $Gini = 1 - \sum_{k} (p_k)^2$，其中 $p_k$ 是类别 $k$ 的样本比例。
*   Gini = 0 表示完全纯净（所有样本属于同一类）。
*   Gini = 0.5 表示最混乱（二分类情况下，两类样本各占一半）。

**目标:** 找到一个分割（一个特征 + 一个阈值），使得分割后的**加权平均 Gini 不纯度最小**，也就是**信息增益最大**。

**信息增益 = (分割前的 Gini) - (分割后的加权平均 Gini)**

**任务:** 再次尝试选择特征和分割阈值，观察分割如何影响 Gini 不纯度，并找到使**信息增益**最大的分割。
""")

# --- Impurity Calculation Functions ---
def calculate_gini(y):
    """计算一个节点 (或数据集) 的基尼不纯度"""
    if len(y) == 0:
        return 0
    counts = np.bincount(y) # 计算每个类别的样本数
    proportions = counts / len(y)
    gini = 1 - np.sum(proportions**2)
    return gini

def calculate_weighted_gini(y_left, y_right):
    """计算分割后的加权平均基尼不纯度"""
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0

    gini_left = calculate_gini(y_left)
    gini_right = calculate_gini(y_right)

    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    return weighted_gini

# --- Stage 3 Interactive Elements ---

# 3.1 计算初始 Gini 不纯度
initial_gini = calculate_gini(y_simple)
st.subheader(f"初始状态 (未分割)")
st.metric(label="整体 Gini 不纯度", value=f"{initial_gini:.4f}")
st.markdown("这个值衡量了开始时数据混合的程度。")


# 3.2 互动控件和结果展示
col3_1, col3_2 = st.columns([1, 2])

with col3_1:
    st.subheader("再次选择分割规则")
    # 重用阶段1的控件变量名，但这里的操作是独立的
    feature_map_s3 = {"特征 X1 (垂直线)": 0, "特征 X2 (水平线)": 1}
    selected_feature_name_s3 = st.radio("选择要依据的特征:", list(feature_map_s3.keys()), key="s3_feature") # key 避免和 stage 1 冲突
    selected_feature_idx_s3 = feature_map_s3[selected_feature_name_s3]

    min_val_s3 = X_simple[:, selected_feature_idx_s3].min()
    max_val_s3 = X_simple[:, selected_feature_idx_s3].max()
    step_s3 = (max_val_s3 - min_val_s3) / 50
    # 使用一个稍微不同的默认值或让用户选择
    default_val_s3 = np.median(X_simple[:, selected_feature_idx_s3]) # 用中位数作为默认值

    split_value_s3 = st.slider(f"设置 '{selected_feature_name_s3.split(' ')[1]}' 的分割阈值:",
                               min_value=min_val_s3, max_value=max_val_s3, value=default_val_s3, step=step_s3, key="s3_slider")


    # 3.3 根据用户的分割进行计算
    # 分割数据
    if selected_feature_idx_s3 == 0: # Split on X1
        left_indices = X_simple[:, 0] <= split_value_s3
        right_indices = X_simple[:, 0] > split_value_s3
    else: # Split on X2
        left_indices = X_simple[:, 1] <= split_value_s3
        right_indices = X_simple[:, 1] > split_value_s3

    y_left = y_simple[left_indices]
    y_right = y_simple[right_indices]

    # 计算 Gini
    gini_left = calculate_gini(y_left)
    gini_right = calculate_gini(y_right)
    weighted_gini_after_split = calculate_weighted_gini(y_left, y_right)
    information_gain = initial_gini - weighted_gini_after_split

    st.subheader("分割后的 Gini 不纯度")
    st.markdown(f"**左侧子集 (<= {split_value_s3:.2f})**")
    st.metric(label=f"样本数: {len(y_left)}", value=f"Gini: {gini_left:.4f}")
    if len(y_left) > 0:
        counts_left = np.bincount(y_left, minlength=2) # minlength 确保总是有两类
        st.caption(f"红🔵: {counts_left[0]}, 蓝🟥: {counts_left[1]}")

    st.markdown(f"**右侧子集 (> {split_value_s3:.2f})**")
    st.metric(label=f"样本数: {len(y_right)}", value=f"Gini: {gini_right:.4f}")
    if len(y_right) > 0:
        counts_right = np.bincount(y_right, minlength=2)
        st.caption(f"红🔵: {counts_right[0]}, 蓝🟥: {counts_right[1]}")

    st.subheader("总体评估")
    st.metric(label="分割后的加权平均 Gini", value=f"{weighted_gini_after_split:.4f}")
    st.metric(label="信息增益 (Gini 减少量)", value=f"{information_gain:.4f}",
              delta=f"{information_gain - 0:.4f}", # 显示增益值本身作为 delta
              help="值越大，表示这次分割带来的“纯度提升”越多。决策树会选择信息增益最大的分割。")


with col3_2:
    st.subheader("数据与当前分割线")
    fig3, ax3 = plot_data(X_simple, y_simple,
                          split_feature=selected_feature_idx_s3,
                          split_value=split_value_s3,
                          title=f"当前分割 (信息增益: {information_gain:.3f})")
    st.pyplot(fig3)

st.markdown("""
**动手试试:**
*   拖动滑块，改变分割阈值。观察左右子集的 Gini 值、加权平均 Gini 和信息增益如何变化。
*   切换选择的特征（X1 或 X2）。
*   你能找到哪个特征和哪个阈值组合，能让**信息增益**达到最大吗？这个组合就是决策树（在第一步）会选择的最佳分割！
""")
st.markdown("---")

# --- 更新后续阶段占位符 ---
# --- Stage 4: 递归构建 - 分而治之 ---
st.header("阶段 4: 递归构建 - 分而治之")
st.markdown("""
我们已经知道如何评估一次分割的好坏（阶段 3）。决策树的构建过程就是**重复**这个寻找“最佳分割”的步骤。

1.  对当前数据集，找到**信息增益最大**的那个分割（特征 + 阈值）。
2.  根据这个分割，将数据集分成两个（或多个）**子集**。
3.  对**每个子集**，**重复步骤 1 和 2**。

这个重复的过程叫做“**递归**”，就像剥洋葱一样，一层一层地处理数据。

**这个过程什么时候停止呢？** 当满足以下任一条件时，就不再对一个子集进行分割，该子集成为一个**叶节点**：
*   **纯净节点:** 该子集里的所有样本都属于同一个类别。 (Gini = 0)
*   **最小样本数:** 该子集的样本数量少于预设的阈值 (例如 `min_samples_leaf`)。
*   **最大深度:** 树的层数已经达到预设的最大深度 (例如 `max_depth`)。
*   **无法再提升纯度:** 找不到任何分割能进一步降低 Gini 不纯度（信息增益 <= 0）。
""")

# --- Function to Find the Best Split ---
def find_best_split(X, y):
    """
    在给定数据集上找到最佳分割点（最大化信息增益）
    返回: best_feature_idx, best_threshold, max_info_gain
    """
    n_samples, n_features = X.shape
    if n_samples <= 1: # 如果样本太少，无法分割
        return None, None, -1

    current_gini = calculate_gini(y)
    if current_gini == 0: # 如果节点已经纯净，无需分割
         return None, None, -1

    max_info_gain = -1 # 初始化为负数
    best_feature_idx = None
    best_threshold = None

    for feature_idx in range(n_features):
        # 尝试所有可能的阈值：特征值排序后，相邻不同值的中点
        thresholds = np.unique(X[:, feature_idx])
        if len(thresholds) > 1:
            potential_thresholds = (thresholds[:-1] + thresholds[1:]) / 2
        else:
            potential_thresholds = thresholds # 只有一个值，没法分割

        for threshold in potential_thresholds:
            # 分割数据
            left_indices = X[:, feature_idx] <= threshold
            right_indices = X[:, feature_idx] > threshold

            y_left = y[left_indices]
            y_right = y[right_indices]

            # 如果分割导致一个子集为空，则跳过这个阈值
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # 计算这次分割的信息增益
            weighted_gini = calculate_weighted_gini(y_left, y_right)
            info_gain = current_gini - weighted_gini

            # 更新最佳分割
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature_idx = feature_idx
                best_threshold = threshold

    # 只有当信息增益大于0时，才认为找到了有效的分割
    if max_info_gain > 0:
        return best_feature_idx, best_threshold, max_info_gain
    else:
        return None, None, -1 # 表示找不到好的分割


# --- 4.1 Apply Best Split to Initial Data ---
st.subheader("4.1 算法找到的第一个最佳分割")

best_feature_idx_s4, best_threshold_s4, max_info_gain_s4 = find_best_split(X_simple, y_simple)

if best_feature_idx_s4 is not None:
    st.success(f"算法找到的最佳初始分割:")
    st.write(f"- **特征:** X{best_feature_idx_s4 + 1}")
    st.write(f"- **阈值:** {best_threshold_s4:.4f}")
    st.write(f"- **最大信息增益:** {max_info_gain_s4:.4f}")

    col4_1a, col4_1b = st.columns(2)
    with col4_1a:
        fig4a, ax4a = plot_data(X_simple, y_simple,
                                split_feature=best_feature_idx_s4,
                                split_value=best_threshold_s4,
                                title="第一个最佳分割线")
        st.pyplot(fig4a)

    with col4_1b:
        # 生成对应的 1 层决策树图
        dot_tree_s4a = graphviz.Digraph()
        node_text = f"X{best_feature_idx_s4 + 1} <= {best_threshold_s4:.2f} ?\nGini={initial_gini:.3f}\nSamples={len(y_simple)}"
        dot_tree_s4a.node('0', node_text)

        # 分割数据以计算子节点信息 (仅用于显示)
        left_indices_s4 = X_simple[:, best_feature_idx_s4] <= best_threshold_s4
        right_indices_s4 = X_simple[:, best_feature_idx_s4] > best_threshold_s4
        y_left_s4 = y_simple[left_indices_s4]
        y_right_s4 = y_simple[right_indices_s4]
        gini_left_s4 = calculate_gini(y_left_s4)
        gini_right_s4 = calculate_gini(y_right_s4)
        pred_left = np.argmax(np.bincount(y_left_s4)) if len(y_left_s4)>0 else -1
        pred_right = np.argmax(np.bincount(y_right_s4)) if len(y_right_s4)>0 else -1
        class_labels = {0: "红🔵", 1: "蓝🟥", -1: "空"}


        node_left_text = f"Gini={gini_left_s4:.3f}\nSamples={len(y_left_s4)}\nPred: {class_labels[pred_left]}"
        node_right_text = f"Gini={gini_right_s4:.3f}\nSamples={len(y_right_s4)}\nPred: {class_labels[pred_right]}"

        dot_tree_s4a.node('1', node_left_text)
        dot_tree_s4a.node('2', node_right_text)
        dot_tree_s4a.edge('0', '1', label='是 (True)')
        dot_tree_s4a.edge('0', '2', label='否 (False)')
        st.graphviz_chart(dot_tree_s4a)

else:
    st.warning("在此数据集上找不到有效的初始分割。")


# --- 4.2 Explore Splitting a Subset ---
st.subheader("4.2 对子集重复寻找最佳分割")
st.markdown("""
现在数据被分成了两个子集（对应上面树图的两个椭圆）。决策树会对**每个**纯度不为0（Gini > 0）的子集，**重复**寻找最佳分割的过程。

让我们选择其中一个子集，看看算法会如何继续分割它：
""")

# 让用户选择要进一步分割的子集
if best_feature_idx_s4 is not None: # 只有在找到第一个分割后才进行
    subset_choice = st.radio("选择要进一步分析的子集:",
                             (f"左子集 (X{best_feature_idx_s4 + 1} <= {best_threshold_s4:.2f})",
                              f"右子集 (X{best_feature_idx_s4 + 1} > {best_threshold_s4:.2f})"),
                             key="subset_choice")

    if subset_choice.startswith("左子集"):
        X_subset = X_simple[left_indices_s4]
        y_subset = y_simple[left_indices_s4]
        parent_node_id = '1' # 对应上面树图的左节点 ID
        st.markdown(f"当前分析: **左子集** (包含 {len(y_subset)} 个样本)")
    else:
        X_subset = X_simple[right_indices_s4]
        y_subset = y_simple[right_indices_s4]
        parent_node_id = '2' # 对应上面树图的右节点 ID
        st.markdown(f"当前分析: **右子集** (包含 {len(y_subset)} 个样本)")

    # 对选定的子集寻找最佳分割
    best_feature_idx_sub, best_threshold_sub, max_info_gain_sub = find_best_split(X_subset, y_subset)

    if best_feature_idx_sub is not None:
        st.success(f"算法找到该子集的最佳分割:")
        st.write(f"- **特征:** X{best_feature_idx_sub + 1}")
        st.write(f"- **阈值:** {best_threshold_sub:.4f}")
        st.write(f"- **信息增益 (相对于此子集):** {max_info_gain_sub:.4f}")

        col4_2a, col4_2b = st.columns(2)
        with col4_2a:
            # 仅绘制子集数据和其分割线
            fig4b, ax4b = plot_data(X_subset, y_subset,
                                    split_feature=best_feature_idx_sub,
                                    split_value=best_threshold_sub,
                                    title="子集内的最佳分割线")
            st.pyplot(fig4b)

        with col4_2b:
            st.markdown("**决策树生长:**")
            # 复制基础树结构
            dot_tree_s4b = dot_tree_s4a.copy()
            # 添加新的层级
            new_node_id_base = parent_node_id # '1' or '2'
            new_node_text = f"X{best_feature_idx_sub + 1} <= {best_threshold_sub:.2f} ?\nGini={calculate_gini(y_subset):.3f}\nSamples={len(y_subset)}"
            # 替换原子集节点为新的内部节点
            dot_tree_s4b.node(new_node_id_base, new_node_text)

            # 分割子集数据以计算叶节点信息
            left_indices_sub = X_subset[:, best_feature_idx_sub] <= best_threshold_sub
            right_indices_sub = X_subset[:, best_feature_idx_sub] > best_threshold_sub
            y_left_sub = y_subset[left_indices_sub]
            y_right_sub = y_subset[right_indices_sub]
            gini_left_sub = calculate_gini(y_left_sub)
            gini_right_sub = calculate_gini(y_right_sub)
            pred_left_sub = np.argmax(np.bincount(y_left_sub)) if len(y_left_sub)>0 else -1
            pred_right_sub = np.argmax(np.bincount(y_right_sub)) if len(y_right_sub)>0 else -1

            # 创建新的叶节点
            new_leaf_left_id = new_node_id_base + 'L' # e.g., '1L' or '2L'
            new_leaf_right_id = new_node_id_base + 'R' # e.g., '1R' or '2R'
            node_left_sub_text = f"Gini={gini_left_sub:.3f}\nSamples={len(y_left_sub)}\nPred: {class_labels[pred_left_sub]}"
            node_right_sub_text = f"Gini={gini_right_sub:.3f}\nSamples={len(y_right_sub)}\nPred: {class_labels[pred_right_sub]}"
            dot_tree_s4b.node(new_leaf_left_id, node_left_sub_text)
            dot_tree_s4b.node(new_leaf_right_id, node_right_sub_text)
            dot_tree_s4b.edge(new_node_id_base, new_leaf_left_id, label='是 (True)')
            dot_tree_s4b.edge(new_node_id_base, new_leaf_right_id, label='否 (False)')

            st.graphviz_chart(dot_tree_s4b)
            st.caption("观察决策树如何在选定的分支下增加了新的节点。")

    else:
        gini_subset = calculate_gini(y_subset)
        if gini_subset == 0:
            st.info(f"该子集已经**纯净** (Gini = {gini_subset:.3f})，无需再分割，成为叶节点。")
            # 可以只显示子集数据点，不画分割线
            fig4b_pure, ax4b_pure = plot_data(X_subset, y_subset, title="纯净的子集")
            st.pyplot(fig4b_pure)
        elif len(y_subset) <= 1: # 示例：添加一个最小样本数的停止条件
             st.info(f"该子集样本数 ({len(y_subset)}) 过少，停止分割，成为叶节点。")
             fig4b_small, ax4b_small = plot_data(X_subset, y_subset, title="样本过少的子集")
             st.pyplot(fig4b_small)
        else:
            st.warning(f"在此子集上找不到信息增益大于 0 的有效分割 (当前 Gini = {gini_subset:.3f})。该子集成为叶节点。")
            fig4b_nosplit, ax4b_nosplit = plot_data(X_subset, y_subset, title="无法有效分割的子集")
            st.pyplot(fig4b_nosplit)


st.markdown("""
**理解关键点:**
*   决策树构建是一个**递归**过程，不断地对产生的子集应用“寻找最佳分割”的逻辑。
*   这个过程会持续下去，直到满足**停止条件**（节点纯净、样本太少、达到最大深度等），这时就形成了树的叶子。
*   整个过程的目标是逐步降低不纯度，提高分类的准确性。
""")
st.markdown("---")
# --- Stage 5: 过拟合的陷阱与超参数的缰绳 ---
st.header("Stage 5: 过拟合的陷阱与超参数的缰绳")
st.markdown("""
我们已经了解了决策树是如何一步步构建的。现在，让我们看看如果让算法**自由生长**（不加限制），会发生什么。
我们会使用 Scikit-learn 库来自动构建树。
""")

# --- 5.1 演示过拟合 ---
st.subheader("5.1 “自由生长”的决策树：过拟合演示")
st.markdown("""
下面的决策树是在我们之前的简单二维数据集上训练的，但是**没有设置最大深度 (`max_depth`) 或叶节点最小样本数 (`min_samples_leaf`) 的限制**。
观察它的结构和决策边界：
""")

col5_1_vis, col5_1_exp = st.columns([2, 1]) # 可视化区域宽，解释区域窄

with col5_1_vis:
    # 训练一个“完全生长”的树
    try:
        clf_overfit = DecisionTreeClassifier(
            criterion='gini', # 可以选择 gini 或 entropy
            random_state=42,
            max_depth=None, # 不限制深度
            min_samples_leaf=1 # 允许叶子只有1个样本
        )
        clf_overfit.fit(X_simple, y_simple) # 在简单数据集上训练

        # 显示树结构
        st.markdown("**决策树结构图 (可能非常复杂)**")
        dot_data_overfit = export_graphviz(clf_overfit, out_file=None,
                                          feature_names=['X1', 'X2'], # 简单特征名
                                          class_names=['红🔵', '蓝🟥'],
                                          filled=True, rounded=True,
                                          special_characters=True)
        st.graphviz_chart(dot_data_overfit)
        acc_overfit = accuracy_score(y_simple, clf_overfit.predict(X_simple))
        st.caption(f"模型在训练集上的准确率: {acc_overfit:.2%}")


        # 绘制决策边界
        st.markdown("**决策边界图 (可能非常曲折)**")
        fig5_overfit, ax5_overfit = plt.subplots(figsize=(7, 6))

        x_min_of, x_max_of = X_simple[:, 0].min() - 0.5, X_simple[:, 0].max() + 0.5
        y_min_of, y_max_of = X_simple[:, 1].min() - 0.5, X_simple[:, 1].max() + 0.5
        h_of = 0.02
        xx_of, yy_of = np.meshgrid(np.arange(x_min_of, x_max_of, h_of), np.arange(y_min_of, y_max_of, h_of))

        Z_of = clf_overfit.predict(np.c_[xx_of.ravel(), yy_of.ravel()])
        Z_of = Z_of.reshape(xx_of.shape)

        cmap_light_of = plt.cm.RdYlBu
        ax5_overfit.contourf(xx_of, yy_of, Z_of, cmap=cmap_light_of, alpha=0.6)

        colors_simple = ['red', 'blue']
        markers_simple = ['o', 's']
        for cl in np.unique(y_simple):
            ax5_overfit.scatter(X_simple[y_simple==cl, 0], X_simple[y_simple==cl, 1],
                               c=colors_simple[cl], marker=markers_simple[cl], edgecolor='k', s=50, label=f'类别 {cl}')

        ax5_overfit.set_xlabel("特征 X1")
        ax5_overfit.set_ylabel("特征 X2")
        ax5_overfit.set_title("自由生长树的决策边界")
        ax5_overfit.legend()
        ax5_overfit.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig5_overfit)

    except Exception as e:
        st.error(f"构建或可视化自由生长树时出错: {e}")


with col5_1_exp:
    st.warning("**观察到了吗？**")
    st.markdown("""
    *   树的结构可能变得非常深、非常复杂，有很多层和很多叶子节点。
    *   决策边界变得非常**弯曲和不规则**，它试图完美地包围训练数据中的每一个点，甚至是那些看起来像“噪音”的点（比如混在对方颜色区域里的点）。
    *   虽然它在训练数据上的准确率可能很高（甚至100%），但这种过于复杂的模型很可能无法很好地适应**新的、未见过的数据**。我们称这种现象为“**过拟合 (Overfitting)**”。
    """)


# --- 5.2 引入超参数控制 ---
st.subheader("5.2 使用超参数控制复杂度")
st.markdown("""
为了防止过拟合，我们需要给决策树的生长加上限制。就像给马套上缰绳，我们可以使用**超参数 (Hyperparameters)** 来控制模型的复杂度。

**尝试调整下面的超参数，观察决策树结构和决策边界如何变化：**
""")

col5_2_params, col5_2_vis = st.columns([1, 2]) # 参数栏窄，可视化区域宽

with col5_2_params:
    st.markdown("**限制条件:**")
    # 超参数控件
    max_depth_s5_ctrl = st.slider(
        "最大深度 (max_depth): 限制树的最大层数",
        min_value=1, max_value=15, value=3, step=1, key="s5_ctrl_max_depth",
        help="较小的值使树更简单，防止过拟合。None表示不限制。"
    )
    min_samples_leaf_s5_ctrl = st.slider(
        "叶节点最小样本数 (min_samples_leaf): 叶子节点最少包含的样本数",
        min_value=1, max_value=len(X_simple)//2, value=1, step=1, key="s5_ctrl_min_leaf", # 最大不超过总样本一半
        help="较大的值防止树分得过细，使模型更稳定。"
    )
    # 可选: 增加 criterion 控制
    # criterion_s5_ctrl = st.radio("分裂标准 (criterion)", ('gini', 'entropy'), key="s5_ctrl_criterion")


with col5_2_vis:
    # 根据用户选择的超参数重新训练模型
    try:
        clf_controlled = DecisionTreeClassifier(
            criterion='gini', # 使用上面选的 criterion_s5_ctrl 如果添加了该控件
            random_state=42,
            max_depth=max_depth_s5_ctrl if max_depth_s5_ctrl > 0 else None, # slider 最小值是 1，所以可以直接用
            min_samples_leaf=min_samples_leaf_s5_ctrl
        )
        clf_controlled.fit(X_simple, y_simple)

        # 显示受控树的结构
        st.markdown("**受控决策树结构图**")
        dot_data_ctrl = export_graphviz(clf_controlled, out_file=None,
                                        feature_names=['X1', 'X2'],
                                        class_names=['红🔵', '蓝🟥'],
                                        filled=True, rounded=True,
                                        special_characters=True)
        st.graphviz_chart(dot_data_ctrl)
        acc_controlled = accuracy_score(y_simple, clf_controlled.predict(X_simple))
        st.caption(f"当前模型在训练集上的准确率: {acc_controlled:.2%}")


        # 绘制受控树的决策边界
        st.markdown("**受控决策树边界图**")
        fig5_ctrl, ax5_ctrl = plt.subplots(figsize=(7, 6))

        # 重用之前的网格和颜色映射
        Z_ctrl = clf_controlled.predict(np.c_[xx_of.ravel(), yy_of.ravel()])
        Z_ctrl = Z_ctrl.reshape(xx_of.shape)

        ax5_ctrl.contourf(xx_of, yy_of, Z_ctrl, cmap=cmap_light_of, alpha=0.6)

        for cl in np.unique(y_simple): # 重绘数据点
            ax5_ctrl.scatter(X_simple[y_simple==cl, 0], X_simple[y_simple==cl, 1],
                             c=colors_simple[cl], marker=markers_simple[cl], edgecolor='k', s=50, label=f'类别 {cl}')

        ax5_ctrl.set_xlabel("特征 X1")
        ax5_ctrl.set_ylabel("特征 X2")
        ax5_ctrl.set_title(f"受控树边界 (depth={max_depth_s5_ctrl}, min_leaf={min_samples_leaf_s5_ctrl})")
        ax5_ctrl.legend()
        ax5_ctrl.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig5_ctrl)

    except Exception as e:
        st.error(f"构建或可视化受控树时出错: {e}")

st.markdown("""
**思考与总结:**
*   比较“自由生长”的树和“受控”的树，它们在结构复杂度和决策边界平滑度上有何不同？
*   `max_depth` 如何影响树的大小和边界？ 值越小，树越简单，边界越趋向于直线或简单的阶梯状。
*   `min_samples_leaf` 如何影响树的大小和边界？ 值越大，树越不容易产生那些只针对少数几个点的细小分支，边界也可能更平滑。
*   通过调整这些超参数，我们可以找到一个在“拟合训练数据”（可能导致过拟合）和“保持模型简单以适应新数据”（可能导致欠拟合）之间的**平衡点**。这在机器学习中称为**模型选择**或**超参数调优**。
""")
st.markdown("---")


# --- Stage 6: 应用于 Iris 数据集 ---
st.header("Stage 6: 应用于 Iris 数据集")
st.markdown("""
现在我们已经理解了过拟合以及如何用超参数控制它。让我们在一个更真实、稍复杂的数据集——**鸢尾花 (Iris)** 上，应用这些知识。

**任务:** 像刚才一样，调整超参数，观察在 Iris 数据集上生成的决策树结构和二维决策边界。注意 Iris 数据有 3 个类别。
""")

# --- 加载 Iris 数据并创建 DataFrame ---
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names_iris = iris.feature_names
target_names_iris = iris.target_names
df_iris = pd.DataFrame(data=X_iris, columns=feature_names_iris)
# 可选: 添加目标列
# df_iris['target'] = y_iris
# df_iris['species'] = pd.Categorical.from_codes(y_iris, target_names_iris)

st.subheader("鸢尾花 (Iris) 数据集回顾")
st.dataframe(df_iris.head(3)) # 显示少量数据

# --- 5.2 超参数控制 (针对 Iris) ---
st.subheader("调整超参数并观察 Iris 数据结果")

col6_params, col6_vis = st.columns([1, 3])

with col6_params:
    st.markdown("**控制树的复杂度:**")
    # 使用新的 key 以免冲突
    max_depth_s6 = st.slider("最大深度 (max_depth)", min_value=1, max_value=10, value=3, step=1, key="s6_max_depth")
    min_samples_leaf_s6 = st.slider("叶节点最小样本数 (min_samples_leaf)", min_value=1, max_value=20, value=1, step=1, key="s6_min_leaf")
    criterion_s6 = st.radio("分裂标准 (criterion)", ('gini', 'entropy'), key="s6_criterion")

    st.markdown("**选择2D可视化特征:**")
    x_feature_idx_s6 = st.selectbox("X轴特征", range(len(feature_names_iris)), format_func=lambda i: feature_names_iris[i], index=2, key="s6_x_feature")
    y_feature_idx_s6 = st.selectbox("Y轴特征", range(len(feature_names_iris)), format_func=lambda i: feature_names_iris[i], index=3, key="s6_y_feature")

    if x_feature_idx_s6 == y_feature_idx_s6:
        st.warning("请为X轴和Y轴选择不同的特征。")
        st.stop()

# --- 5.3 训练模型与可视化 (针对 Iris) ---
with col6_vis:
    # 1. 训练完整模型 (Iris)
    try:
        clf_iris_full_s6 = DecisionTreeClassifier(
            max_depth=max_depth_s6,
            min_samples_leaf=min_samples_leaf_s6,
            criterion=criterion_s6,
            random_state=42
        )
        clf_iris_full_s6.fit(X_iris, y_iris)

        # 2. 生成树结构图 (Iris)
        st.markdown("**决策树结构图 (基于全部4个特征)**")
        dot_data_iris_s6 = export_graphviz(clf_iris_full_s6, out_file=None,
                                          feature_names=feature_names_iris,
                                          class_names=target_names_iris,
                                          filled=True, rounded=True,
                                          special_characters=True)
        st.graphviz_chart(dot_data_iris_s6)
        accuracy_iris_s6 = accuracy_score(y_iris, clf_iris_full_s6.predict(X_iris))
        st.caption(f"当前模型在训练集上的准确率: {accuracy_iris_s6:.2%}")

    except Exception as e:
        st.error(f"无法构建或显示 Iris 决策树结构图。错误: {e}")

    # 3. 训练 2D 模型 (Iris)
    try:
        X_iris_2d_s6 = X_iris[:, [x_feature_idx_s6, y_feature_idx_s6]]
        clf_iris_2d_s6 = DecisionTreeClassifier(
            max_depth=max_depth_s6,
            min_samples_leaf=min_samples_leaf_s6,
            criterion=criterion_s6,
            random_state=42
        )
        clf_iris_2d_s6.fit(X_iris_2d_s6, y_iris)

        # 4. 绘制决策边界 (Iris)
        st.markdown(f"**决策边界图 (基于 '{feature_names_iris[x_feature_idx_s6]}' 和 '{feature_names_iris[y_feature_idx_s6]}')**")
        fig6, ax6 = plt.subplots(figsize=(8, 6))

        x_min_i, x_max_i = X_iris_2d_s6[:, 0].min() - 0.5, X_iris_2d_s6[:, 0].max() + 0.5
        y_min_i, y_max_i = X_iris_2d_s6[:, 1].min() - 0.5, X_iris_2d_s6[:, 1].max() + 0.5
        h_i = 0.02
        xx_i, yy_i = np.meshgrid(np.arange(x_min_i, x_max_i, h_i), np.arange(y_min_i, y_max_i, h_i))

        Z_i = clf_iris_2d_s6.predict(np.c_[xx_i.ravel(), yy_i.ravel()])
        Z_i = Z_i.reshape(xx_i.shape)

        cmap_light_i = plt.cm.RdYlBu
        ax6.contourf(xx_i, yy_i, Z_i, cmap=cmap_light_i, alpha=0.6)

        cmap_bold_i = plt.cm.viridis
        scatter_i = ax6.scatter(X_iris_2d_s6[:, 0], X_iris_2d_s6[:, 1], c=y_iris, cmap=cmap_bold_i,
                                edgecolor='k', s=40)

        ax6.set_xlabel(feature_names_iris[x_feature_idx_s6])
        ax6.set_ylabel(feature_names_iris[y_feature_idx_s6])
        ax6.set_title("Iris 数据集决策边界")
        handles_i, _ = scatter_i.legend_elements(prop="colors")
        ax6.legend(handles_i, target_names_iris, title="类别")
        ax6.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig6)

    except Exception as e:
        st.error(f"无法绘制 Iris 决策边界图。错误: {e}")

st.markdown("---")


# --- Stage 7: 总结与应用 (原 Stage 6) ---
st.header("Stage 7: 总结与应用")

st.markdown("""
恭喜你完成了决策树探秘之旅！让我们回顾一下核心要点：

1.  **是什么？** 决策树是一种监督学习算法，通过学习一系列基于特征的“是/否”问题（规则），来对数据进行分类或回归。它像一个流程图。
2.  **如何构建？**
    *   **选择最佳分割:** 在每个节点，算法会尝试所有可能的特征和阈值，找到能最大程度“纯化”数据（例如，最大化信息增益或最小化基尼不纯度）的分割。
    *   **递归分裂:** 对分割产生的子集重复寻找最佳分割的过程，直到满足停止条件。
    *   **停止条件:** 节点纯净、样本数过少、达到最大深度等，防止树无限生长。
3.  **过拟合与控制？**
    *   **过拟合:** 决策树容易过度学习训练数据的细节和噪声，导致在新数据上表现不佳。
    *   **超参数控制:** 使用**超参数**（如 `max_depth`, `min_samples_leaf`）来限制树的复杂度，防止过拟合，提高模型的泛化能力。
4.  **如何使用？** 可以使用 Scikit-learn 等库自动构建和调整决策树模型。

**决策树的优点:**
*   **可解释性强:** 树的结构清晰，容易理解分类规则。
*   **对数据预处理要求低:** 通常不需要特征缩放。
*   **可以处理数值型和类别型特征。**

**决策树的缺点:**
*   **容易过拟合:** 特别是当树很深时。
*   **对数据微小变动敏感:** 数据的小变化可能导致生成完全不同的树（不稳定）。
*   **可能产生有偏树:** 如果某些类别样本量远大于其他类别。

**应用场景:**
决策树本身可用于分类和回归任务。更重要的是，它是许多更强大的**集成学习算法**的基础模块，例如：
*   **随机森林 (Random Forest):** 构建多棵不同的决策树并综合它们的预测结果，通常更稳定且不易过拟合。
*   **梯度提升决策树 (Gradient Boosting Decision Tree - GBDT, XGBoost, LightGBM):** 逐步构建树来纠正之前树的错误，通常精度很高。

希望这次旅程能帮助你建立对决策树的直观理解！
""")
