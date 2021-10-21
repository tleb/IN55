import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
import math
from PIL import Image
import pkg_resources
from collections import OrderedDict


nb_vert_infos_size = 3


class Window:
    def __init__(self, width, height, title):
        self.Window = None

        if not glfw.init():
            return

        self.Window = glfw.create_window(width, height, title, None, None)

        if not self.Window:
            glfw.terminate()
            return

        glfw.make_context_current(self.Window)

        self.updateProjectionMatrix(width, height)

    def updateProjectionMatrix(self, width, height):
        fov = 60
        aspect_ratio = width / height
        near_clip = 0.1
        far_clip = 100

        # create a perspective matrix
        self.ProjectionMatrix = pyrr.matrix44.create_perspective_projection(
            fov, aspect_ratio, near_clip, far_clip
        )

        glViewport(0, 0, width, height)

    def initViewMatrix(self, eye):
        eye = np.array(eye)
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        self.ViewMatrix = pyrr.matrix44.create_look_at(eye, target, up)

    def render(self, objects):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

        v = self.ViewMatrix
        p = self.ProjectionMatrix
        # vp = np.matmul(v, p)

        while not glfw.window_should_close(self.Window):
            glfw.poll_events()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            w, h = glfw.get_framebuffer_size(self.Window)
            self.updateProjectionMatrix(w, h)

            for o in objects:
                o.updateTRSMatrices()
                o.updateModelMatrix()
                # mvp = np.matmul(o.ModelMatrix, vp)

                vertices = o.updateVertices()

                m = o.ModelMatrix
                o.Shader.draw(m, v, p, vertices)

            glfw.swap_buffers(self.Window)

        self.CloseWindow()

    def CloseWindow(self):
        glfw.terminate()


class Object3D:
    def __init__(self):
        self.translate((0, 0, 0))
        self.scale((1, 1, 1))
        self.R = pyrr.matrix44.create_identity()

    def translate(self, vec):
        self.T = pyrr.matrix44.create_from_translation(vec)

    def scale(self, fac):
        self.S = pyrr.matrix44.create_from_scale(fac)

    def rotate(self, matrot, angle):
        self.R = pyrr.matrix44.create_from_axis_rotation(matrot, angle)

    def updateModelMatrix(self):
        self.ModelMatrix = np.matmul(np.matmul(self.S, self.R), self.T)

    def updateTRSMatrices(self):
        pass

    def updateVertices(self):
        pass


def CreateShader(name):
    vs_file = open(name + ".vs.txt", "r")
    VERTEX_SHADER = vs_file.read()
    vs_file.close()

    fs_file = open(name + ".fs.txt", "r")
    FRAGMENT_SHADER = fs_file.read()
    fs_file.close()

    # Compile The Program and shaders
    return OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
    )


class Texture:
    def __init__(self, path, texture_unit, uniform_name, shader):
        # Path to the image file.
        self.path = path
        # The "texture name", as called in the doc.
        self.texture_name = glGenTextures(1)
        # The texture unit, as called in the doc. Use a constant such as GL_TEXTUREx.
        self.texture_unit = texture_unit
        # The texture name, in the shader.
        self.uniform_name = uniform_name
        self.shader = shader

        image = Image.open(self.path).convert("RGBA")
        width, height = image.size
        image_data = np.array(list(image.getdata()), np.uint8)

        glActiveTexture(self.texture_unit)
        glBindTexture(GL_TEXTURE_2D, self.texture_name)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            width,
            height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            image_data,
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)

    def draw_texture(self):
        glActiveTexture(self.texture_unit)
        glBindTexture(GL_TEXTURE_2D, self.texture_name)
        uniformLoc = glGetUniformLocation(self.shader, self.uniform_name)
        glUniform1i(uniformLoc, int(self.texture_unit) - int(GL_TEXTURE0))


class PositionShader:
    def __init__(
        self, path_prefix, vertices, primitives, textures, color=(1.0, 1.0, 1.0)
    ):
        self.Vertices = vertices
        self.Primitives = primitives
        self.Shader = CreateShader(path_prefix)
        self.Color = color
        self.textures = []
        self.m_loc = None
        self.v_loc = None
        self.p_loc = None
        self.c_loc = None
        self.t_loc = None
        self.textures = textures
        self.createBuffers()

    def createBuffers(self):
        vertices = np.array(self.Vertices, dtype=np.float32)

        self.NbVertices = int(len(vertices) / nb_vert_infos_size)

        # Create Buffer object in gpu
        self.VBO = glGenBuffers(1)
        # Bind the buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.NbVertices * nb_vert_infos_size * 4,
            vertices,
            GL_STATIC_DRAW,
        )

        self.textures = [Texture(x[0], x[1], x[2], self.Shader) for x in self.textures]

    def use(self, vertices):
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)

        if vertices is not None:
            glBufferSubData(
                GL_ARRAY_BUFFER, 0, self.NbVertices * nb_vert_infos_size * 4, vertices
            )

        positionLoc = glGetAttribLocation(self.Shader, "position")
        glVertexAttribPointer(
            positionLoc, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0)
        )
        glEnableVertexAttribArray(positionLoc)

        glUseProgram(self.Shader)

    def draw(self, m, v, p, vertices):
        self.use(vertices)

        for texture in self.textures:
            texture.draw_texture()

        if self.m_loc is None:
            self.m_loc = glGetUniformLocation(self.Shader, "m")
            self.v_loc = glGetUniformLocation(self.Shader, "v")
            self.p_loc = glGetUniformLocation(self.Shader, "p")
            self.c_loc = glGetUniformLocation(self.Shader, "Color")
            self.t_loc = glGetUniformLocation(self.Shader, "t")

        glUniformMatrix4fv(self.m_loc, 1, GL_FALSE, m)
        glUniformMatrix4fv(self.v_loc, 1, GL_FALSE, v)
        glUniformMatrix4fv(self.p_loc, 1, GL_FALSE, p)
        glUniform3fv(self.c_loc, 1, self.Color)
        glUniform1f(self.t_loc, glfw.get_time())

        for p in self.Primitives:
            nb_indices = len(p[1])
            glDrawElements(p[0], nb_indices, GL_UNSIGNED_INT, p[1])


def addVertex(tab, p, c):
    tab.extend(p)
    tab.extend(c)
