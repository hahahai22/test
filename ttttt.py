import numpy as np
from stl import mesh
import math
import os

class BlueAndWhiteVase:
    def __init__(self):
        self.points = 100  # 圆周上的点数
        self.sections = 50  # 高度方向的分段数
        
    def vase_profile(self, t):
        """定义花瓶的轮廓曲线"""
        # 可以根据需要调整这些参数来改变花瓶的形状
        base_radius = 20
        neck_height = 0.7
        neck_width = 0.5
        belly_height = 0.4
        belly_width = 1.2
        
        if t < neck_height:
            # 瓶颈部分
            return base_radius * (neck_width + (1 - neck_width) * (1 - t/neck_height))
        else:
            # 瓶身部分
            return base_radius * (belly_width - (belly_width - 1) * ((t - neck_height)/(1 - neck_height))**2)
    
    def create_vase(self, height=100, filename="blue_and_white_vase.stl"):
        """创建青花瓷瓶模型"""
        # 生成花瓶表面的点
        vertices = []
        for i in range(self.sections + 1):
            t = i / self.sections
            z = height * t
            radius = self.vase_profile(t)
            
            for j in range(self.points):
                angle = 2 * math.pi * j / self.points
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                vertices.append([x, y, z])
        
        # 创建底部顶点
        vertices.append([0, 0, 0])
        bottom_center_index = len(vertices) - 1
        
        # 创建顶部顶点（开口处）
        vertices.append([0, 0, height])
        top_center_index = len(vertices) - 1
        
        # 创建面
        faces = []
        
        # 创建侧面
        for i in range(self.sections):
            for j in range(self.points):
                current = i * self.points + j
                next_row = (i + 1) * self.points + j
                next_point = i * self.points + (j + 1) % self.points
                next_row_next_point = (i + 1) * self.points + (j + 1) % self.points
                
                faces.append([current, next_row, next_point])
                faces.append([next_row, next_row_next_point, next_point])
        
        # 创建底部
        for j in range(self.points):
            current = j
            next_point = (j + 1) % self.points
            faces.append([bottom_center_index, current, next_point])
        
        # 创建顶部（开口处）
        for j in range(self.points):
            current = (self.sections - 1) * self.points + j
            next_point = (self.sections - 1) * self.points + (j + 1) % self.points
            faces.append([top_center_index, next_point, current])
        
        # 创建网格
        vase = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                vase.vectors[i][j] = vertices[f[j]]
        
        # 保存为STL文件
        vase.save(filename)
        print(f"当前工作目录: {os.getcwd()}")
        print(f"顶点数量: {len(vertices)}")
        print(f"面片数量: {len(faces)}")
        print(f"青花瓷瓶已保存为: {filename}")
        return filename
    
    def add_patterns(self, vase_mesh, pattern_type="flowers", density=0.5):
        """为花瓶添加图案（目前只是概念，需要进一步实现）"""
        # 这个函数需要更复杂的实现来真正添加图案
        # 可能需要使用雕刻或表面修改技术
        pass

if __name__ == "__main__":
    vase_printer = BlueAndWhiteVase()
    
    # 生成青花瓷瓶
    vase_file = vase_printer.create_vase()
    print(f"青花瓷瓶已保存为: {vase_file}")
    
    # 可以添加图案（目前仅为概念）
    # vase_printer.add_patterns(vase_mesh)
