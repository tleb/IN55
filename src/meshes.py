import numpy as np
import time
import stl
import numpy.linalg as LA
import math
import random as rd
from opengl_fcts import *


class Soucoupe(Object3D):
    def __init__(self):
        super().__init__()

        self.previous_time = time.time()

        mesh = stl.Mesh.from_file("../soucoupe.stl")
        assert mesh.points.shape[1] == 9

        vertices = mesh.points.reshape((mesh.points.size // 3, 3))

        print("vertex count:", vertices.shape[0])

        indices = np.array((range(vertices.size)))

        vertices = vertices.reshape(vertices.size)
        primitives = [(GL_TRIANGLES, indices)]
        self.Shader = PositionShader(
            "soucoupe",
            vertices,
            primitives,
            [("./texture-metal.jpg", GL_TEXTURE1, "tex")],
        )

    def updateTRSMatrices(self):
        current_time = time.time()
        print(int((current_time - self.previous_time) * 1000), "ms")
        self.previous_time = current_time

        t = glfw.get_time()
        trans_x = pyrr.matrix44.create_from_translation(
            [
                -15 * (np.sin((t + np.pi) / 2) + 1),
                1.0,
                -15 * (np.sin((t + np.pi) / 2) + 1),
            ]
        )

        self.T = trans_x


class Planete(Object3D):
    def __init__(self):
        super().__init__()

        self.previous_time = time.time()

        mesh = stl.Mesh.from_file("../asteroide.stl")
        assert mesh.points.shape[1] == 9

        vertices = mesh.points.reshape((mesh.points.size // 3, 3))

        print("vertex count:", vertices.shape[0])

        indices = np.array(range(vertices.size))

        vertices = vertices.reshape(vertices.size)
        primitives = [(GL_TRIANGLES, indices)]
        self.Shader = PositionShader(
            "asteroide",
            vertices,
            primitives,
            [("./texture-asteroide.jpg", GL_TEXTURE1, "tex")],
        )

    def updateTRSMatrices(self):
        current_time = time.time()
        print(int((current_time - self.previous_time) * 1000), "ms")
        self.previous_time = current_time

        t = glfw.get_time()

        rot_z = pyrr.Matrix44.from_z_rotation(t / 2.0)
        self.R = rot_z


class Asteroide(Object3D):
    def __init__(self):
        super().__init__()

        self.generatevalues()
        self.previous_time = time.time()

        mesh = stl.Mesh.from_file("../asteroide.stl")
        assert mesh.points.shape[1] == 9

        vertices = mesh.points.reshape((mesh.points.size // 3, 3))

        print("vertex count:", vertices.shape[0])

        indices = np.array(range(vertices.size))

        vertices = vertices.reshape(vertices.size)
        primitives = [(GL_TRIANGLES, indices)]
        self.Shader = PositionShader(
            "asteroide",
            vertices,
            primitives,
            [("./texture-asteroide.jpg", GL_TEXTURE1, "tex")],
        )

    def generatevalues(self):
        self.value = (rd.random() * 30) + 80  # Random value between 50 and 100
        self.offset = (rd.random() * 40) - 20  # Random value between -20 and +20
        self.choice1 = rd.choice([-1, 1])
        self.choice2 = rd.choice([-1, 1])
        self.choice3 = rd.choice([-1, 1])
        self.choice4 = rd.choice([-1, 1])

    def updateTRSMatrices(self):
        current_time = time.time()
        print(int((current_time - self.previous_time) * 1000), "ms")
        self.previous_time = current_time

        t = glfw.get_time()

        trans_x = pyrr.matrix44.create_from_translation(
            [
                self.choice1 * self.value * (np.sin(t)) + self.choice2 * self.offset,
                self.choice3 * self.value * (np.sin(t)) + self.choice4 * self.offset,
                -self.value,
            ]
        )

        if (np.sin(t) > 0.999) or (np.sin(t) < -0.999):
            self.generatevalues()

        self.T = trans_x
