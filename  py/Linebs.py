import numpy as np
import matplotlib.pyplot as plt

"""
我们使用 向量、内积、投影、正交 这种抽象概念
解决一个真实问题(简化)

2d环境

me(player), enemy 是 2d坐标坐标系上的的点
我们来模拟射击游戏中的攻击模式

"""

me_pos = np.array([2.0, 2.0])
enemy_pos = np.array([5.0, 3.0])
 
# 算出 me -> ememy 的向量, 使用的 向量减法
v  =  enemy_pos - me_pos

# me 现在是坐标系中的一个点, 但它也会有一个面向方向(也是的一个向量), 它就是子弹的射击方向
shoot_direction =  np.array([1.0, 1.0])
# 归一化 长度为1 保留方向
shoot_direction =  shoot_direction / np.linalg.norm(shoot_direction)

# 到这里整个几何位置关系 会构建一个直角三角形  v 是斜边, shoot_d是底边  enemy_pos 到 shoot_direction的垂直线是 对边
# 我们现在获取第一个构建几何信息 enemy_pos 相对与 me 的方向, 这里我们使用内积
dotp = np.dot(shoot_direction,v) # v 在弹道向量上的投影

# v 的模,实际长度
v_len = np.linalg.norm(v)

# 这里我们知道点积， |shoot_direction| 和 v 我们利用算出角度
# dot = |v| ⋅ |shoot_direction| ⋅ cos\theta 
# dot / |v| = cos\theta 
cos_t = dotp / v_len 

# 反出角度
theta = np.arccos(cos_t)

# 叉积
cross = shoot_direction[0] * v[1] - shoot_direction[1] * v[0]
# 1 顺时针 0 共线  -1 逆时针 
crs = 1 if cross > 0  else  (-1 if cross < 0  else 0)

print(f"{crs}, {theta}")


# # =======================================================
# # 3. 开始使用 matplotlib 绘图
# # =======================================================
# fig, ax = plt.subplots(figsize=(8, 8))

# # 绘制网格背景
# ax.grid(True, linestyle='--', alpha=0.6)

# # 绘制主角和敌人（点）
# ax.scatter(player_pos[0], player_pos[1], color='blue', s=150, zorder=5, label='Me (Player)')
# ax.scatter(enemy_pos[0], enemy_pos[1], color='red', s=150, zorder=5, label='Enemy')

# # 绘制视线方向向量 u (蓝色箭头，长度设为2以方便观察)
# ax.quiver(player_pos[0], player_pos[1], view_dir[0]*2, view_dir[1]*2, 
#           angles='xy', scale_units='xy', scale=1, color='dodgerblue', 
#           width=0.008, label='View Direction (u) x2')

# # 绘制敌人相对向量 v (橙色箭头)
# ax.quiver(player_pos[0], player_pos[1], v[0], v[1], 
#           angles='xy', scale_units='xy', scale=1, color='darkorange', 
#           width=0.006, label='To Enemy Vector (v)')

# # 绘制垂直投影线 (绿色虚线)
# # 从 敌人位置 连线到 投影点
# ax.plot([enemy_pos[0], proj_point[0]], [enemy_pos[1], proj_point[1]], 
#         color='forestgreen', linestyle=':', linewidth=2.5, label='Perpendicular Distance')

# # 绘制投影长度线 (紫色实线)
# # 从 主角位置 连线到 投影点
# ax.plot([player_pos[0], proj_point[0]], [player_pos[1], proj_point[1]], 
#         color='purple', linestyle='-', linewidth=2, label='Projection Length')

# # 标注文字
# ax.text(player_pos[0]-0.4, player_pos[1]-0.4, 'Me (2,2)', fontsize=12, fontweight='bold', color='blue')
# ax.text(enemy_pos[0]+0.1, enemy_pos[1]+0.1, 'Enemy (5,3)', fontsize=12, fontweight='bold', color='red')
# ax.text(proj_point[0]-0.4, proj_point[1]+0.2, f'Proj Point\n({proj_point[0]:.2f}, {proj_point[1]:.2f})', 
#         fontsize=10, color='purple')

# # 在图上显示计算结果信息框
# info_text = (
#     f"Dot Product (Proj): {dot_product:.2f} m\n"
#     f"Perp Distance: {np.sqrt(v_length**2 - dot_product**2):.2f} m\n"
#     f"Turn Angle: {theta_deg:.2f}°\n"
#     f"Turn Direction: {turn_dir}"
# )
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)

# # 限制坐标轴范围与比例，确保几何角度不失真
# ax.set_xlim(0, 7)
# ax.set_ylim(0, 7)
# ax.set_aspect('equal') # 极其重要：确保1:1比例，否则角度会变形
# ax.set_xlabel('X Axis', fontsize=12)
# ax.set_ylabel('Y Axis', fontsize=12)
# ax.set_title('Visualizing Dot Product, Projection and Rotation', fontsize=14, fontweight='bold')
# ax.legend(loc='lower right')

# # 显示图形
# plt.show()