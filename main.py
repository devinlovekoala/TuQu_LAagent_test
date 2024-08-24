import os
import shutil
import json
from manim import *
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, validator
from typing import List
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置manim视频生成路径
config.media_dir = os.path.join(os.getcwd(), 'static')

class MatrixRequest(BaseModel):
    matrix: List[List[float]]

    @validator('matrix')
    def check_matrix(cls, v):
        if not v or not all(v):
            raise ValueError('Matrix must not be empty')
        if len(set(len(row) for row in v)) != 1:
            raise ValueError('All rows in the matrix must have the same length')
        return v


class InvertibleMatrixAnimation(Scene):
    def construct(self, original_matrix, inverse_matrix):
        # 创建原始矩阵
        matrix = Matrix(original_matrix)
        self.play(Write(matrix))

        # 添加原始矩阵的标签
        original_label = MathTex(r"\text{Original Matrix}")
        original_label.next_to(matrix, UP)
        self.play(Write(original_label))

        # 移动原始矩阵到左侧
        self.play(matrix.animate.shift(LEFT * 3))
        self.play(original_label.animate.shift(LEFT * 3))

        # 创建逆矩阵
        inverse = Matrix(inverse_matrix)
        inverse.shift(RIGHT * 3)

        # 添加逆矩阵的标签
        inverse_label = MathTex(r"\text{Inverse Matrix}")
        inverse_label.next_to(inverse, UP)

        # 显示逆矩阵的计算过程
        self.play(Transform(matrix.copy(), inverse), run_time=2)
        self.play(Write(inverse_label))

        # 显示原始矩阵和逆矩阵的乘积
        product = MathTex(r"A", r"A^{-1}", r"=", r"I")
        product.shift(DOWN * 2)
        self.play(Write(product))

        # 暂停一下，让观众有时间观察
        self.wait(2)


class NonInvertibleMatrixAnimation(Scene):
    def construct(self, matrix):
        # 创建原始矩阵
        matrix_mob = Matrix(matrix)
        self.play(Write(matrix_mob))

        # 添加原始矩阵的标签
        original_label = MathTex(r"\text{Non-invertible Matrix}")
        original_label.next_to(matrix_mob, UP)
        self.play(Write(original_label))

        # 计算行列式
        det = np.linalg.det(matrix)
        det_text = MathTex(f"\\text{{det}}(A) = {det:.2f}")
        det_text.next_to(matrix_mob, DOWN)
        self.play(Write(det_text))

        # 解释为什么矩阵不可逆
        explanation = Text("This matrix is not invertible because its determinant is zero.", font_size=24)
        explanation.next_to(det_text, DOWN)
        self.play(Write(explanation))

        # 如果是2x2矩阵，可以可视化线性变换
        if matrix.shape == (2, 2):
            # 创建一个平面
            plane = NumberPlane()
            self.play(Create(plane))

            # 应用线性变换
            matrix_transform = matrix_mob.copy()
            self.play(ApplyMatrix(matrix, plane))
            matrix_transform.next_to(plane, RIGHT)
            self.play(Write(matrix_transform))

            # 解释维度降低
            dimension_text = Text("The transformation reduces the dimension of the space.", font_size=24)
            dimension_text.next_to(plane, DOWN)
            self.play(Write(dimension_text))

        self.wait(2)


def matrix_to_filename(matrix):
    # 将矩阵元素连接成字符串，用下划线分隔
    matrix_str = "_".join(map(str, [item for sublist in matrix for item in sublist]))
    return f"inverse_{matrix_str}.mp4"


@app.get("/api/process")
async def get_calculator():
    with open("index.html", "r", encoding="utf-8") as file:
        content = file.read()
    return HTMLResponse(content=content)


@app.post("/api/process/inverse_matrix")
async def inverse_matrix(request: MatrixRequest):
    try:
        matrix = np.array(request.matrix)
        result = np.linalg.inv(matrix)

        # 生成文件名
        video_filename = matrix_to_filename(request.matrix)
        output_video_dir = os.path.join(config.media_dir, 'videos', '720p30', 'cache', 'InverseMatrix')
        os.makedirs(output_video_dir, exist_ok=True)

        target_video_path = os.path.join(output_video_dir, video_filename)

        # 检查文件是否已存在
        if not os.path.exists(target_video_path):
            # 如果文件不存在，创建动画
            class MatrixInverseAnimation(Scene):
                def construct(self):
                    matrix_mob = Matrix(matrix)
                    self.play(Write(matrix_mob))
                    self.wait()

                    arrow = Arrow(LEFT, RIGHT)
                    self.play(GrowArrow(arrow))

                    inverse_mob = Matrix(result)
                    self.play(Write(inverse_mob))
                    self.wait(2)

            # 渲染动画，直接输出到指定路径
            config.output_file = target_video_path
            config.quality = "medium_quality"
            scene = MatrixInverseAnimation()
            scene.render()

            print(f"Generated new animation file: {target_video_path}")
        else:
            print(f"Using existing animation file: {target_video_path}")

        # 将路径转换为相对路径，并将反斜杠转换为正斜杠
        relative_path = os.path.relpath(target_video_path, start=config.media_dir)
        relative_path = relative_path.replace("\\", "/")

        # 返回视频文件的URL
        return {"result": result.tolist(), "animation_url": f"/static/{relative_path}"}

    except np.linalg.LinAlgError:
        # 处理非可逆矩阵，生成不可逆矩阵的演示动画
        filename = matrix_to_filename(request.matrix)
        output_dir = os.path.join("static", "media", "videos", "720p30")
        static_path = os.path.join(output_dir, filename)

        if not os.path.exists(static_path):
            class NonInvertibleMatrixScene(NonInvertibleMatrixAnimation):
                def construct(self):
                    super().construct(matrix)

            # 确保静态目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 渲染动画，直接输出到指定路径
            config.output_file = static_path
            config.quality = "medium_quality"
            scene = NonInvertibleMatrixScene()
            scene.render()

            print(f"Generated new animation file for non-invertible matrix: {static_path}")
        else:
            print(f"Using existing animation file for non-invertible matrix: {static_path}")

        # 将路径转换为相对路径，并将反斜杠转换为正斜杠
        relative_path = os.path.relpath(static_path, start=config.media_dir)
        relative_path = relative_path.replace("\\", "/")

        return {"detail": "Matrix is not invertible.", "animation_url": f"/static/{relative_path}"}

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"detail": f"An error occurred: {str(e)}"}
        )


@app.post("/api/process/trans_matrix")
async def transpose_matrix(request: MatrixRequest):
    matrix = np.array(request.matrix)
    result = matrix.T
    # 这里可以添加转置矩阵的动画生成逻辑
    return {"result": result.tolist()}


@app.post("/api/process/num_matrix")
async def scalar_multiplication(request: MatrixRequest):
    # 这里需要修改请求模型以包含标量值
    matrix = np.array(request.matrix)
    scalar = 2  # 假设标量为2，实际应该从请求中获取
    result = scalar * matrix
    # 这里可以添加数乘的动画生成逻辑
    return {"result": result.tolist()}


@app.post("/api/process/product_matrix")
async def matrix_multiplication(request: MatrixRequest):
    # 这里需要修改请求模型以包含两个矩阵
    matrix1 = np.array(request.matrix)
    matrix2 = np.array([[1, 0], [0, 1]])  # 假设第二个矩阵为单位矩阵，实际应该从请求中获取
    try:
        result = np.matmul(matrix1, matrix2)
        # 这里可以添加矩阵乘法的动画生成逻辑
        return {"result": result.tolist()}
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"detail": "Matrices cannot be multiplied. Check their dimensions."}
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)