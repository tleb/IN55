import numpy as np
import time
import stl
import numpy.linalg as LA
import math
from opengl_fcts import *


def homothety_matrix(k):
    return np.diag([k, k, k, 1])


# https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
def rotation_matrix(a, x, y, z):
    b = 1 - math.cos(a)
    c = math.cos(a)
    d = math.sin(a)
    return np.matrix(
        [
            [c + x * b, x * y * b - z * d, x * y * b + y * d, 0],
            [y * x * b + z * d, c + y * y * b, y * z * b - x * d, 0],
            [z * x * b - y * d, z * y * b + x * d, c + z * z * b, 0],
            [0, 0, 0, 1],
        ]
    )


def translation_matrix(x, y, z):
    return np.matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


# Inspired (but modified): https://stackoverflow.com/a/36425155/4255615
def line_point_distance(p, a, b):
    assert p.shape == (3,) and a.shape == (3,) and b.shape == (3,)

    ab = b - a
    ap = p - a

    # This is a linear value on where P is along the line. If negative, it is
    # before A. If between zero and norm(AB), it is along the line. If above
    # norm(AB), it is after B.
    ratio = ap.dot(ab)

    if ratio <= 0:
        return LA.norm(ap)

    if ratio >= LA.norm(ab):
        return LA.norm(p - b)

    return LA.norm(np.cross(ab, ap)) / LA.norm(ab)


class RiggedMesh(Object3D):
    def __init__(self, mesh_path, rig_path, shader_path, textures):
        super().__init__()

        mesh = stl.Mesh.from_file(mesh_path)
        assert mesh.points.shape[1] == 9

        # NAIVE APPROACH, not doing anything about unique vertices
        vertices = mesh.points.reshape((mesh.points.size // 3, 3))

        print("vertex count:", vertices.shape[0])

        indices = np.array(range(vertices.size))

        # Now, deal with bones.

        # Center the mesh around (0, 0, 0).
        middle_x = (np.min(vertices[:, 0]) + np.max(vertices[:, 0])) / 2
        middle_y = (np.min(vertices[:, 1]) + np.max(vertices[:, 1])) / 2
        middle_z = (np.min(vertices[:, 2]) + np.max(vertices[:, 2])) / 2
        vertices = vertices - [middle_x, middle_y, middle_z]

        # Import the rig...
        bones = read_rig_export(rig_path)

        # Put bone heads and tails into a NumPy array.
        bone_vertices = np.empty(0)
        for bone in bones.values():
            bone_vertices = np.r_[bone_vertices, bone.head, bone.tail]
        bone_vertices = bone_vertices.reshape((bone_vertices.size // 3, 3))

        # Center the rig around (0, 0, 0).
        middle_x = (np.min(bone_vertices[:, 0]) + np.max(bone_vertices[:, 0])) / 2
        middle_y = (np.min(bone_vertices[:, 1]) + np.max(bone_vertices[:, 1])) / 2
        middle_z = (np.min(bone_vertices[:, 2]) + np.max(bone_vertices[:, 2])) / 2
        bone_vertices = bone_vertices - [middle_x, middle_y, middle_z]
        bone_vertices = bone_vertices.reshape((bone_vertices.size // 6, 6))

        # Scale the mesh based on the Z axis. We should avoid this if we can, or
        # at least find the biggest dimension.
        # rig_z_width = np.max(bone_vertices[:, 2]) - np.min(bone_vertices[:, 2])
        # mesh_z_width = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        # vertices = vertices * (rig_z_width / mesh_z_width)

        # Measure the distance between each vertex and each bone (its center
        # point which is an approximation that can be done using NumPy).
        distances = None
        for index, (bone_name, bone) in enumerate(bones.items()):
            bone.head = bone_vertices[index][0:3]
            bone.tail = bone_vertices[index][3:6]
            center_point = bone.center_point()

            # We actually use distance squared, not distance. This is easier to
            # compute and works the same way.
            dist = (vertices - center_point) ** 2
            dist = np.sum(dist, axis=1)
            if distances is None:
                distances = np.mat(dist).T
            else:
                distances = np.c_[distances, dist.T]

        # This stores the bone to which a vertex is linked.
        vertex_bone_assignment = np.argmin(distances, axis=1).A1
        print("bone usage:", np.unique(vertex_bone_assignment).size, "/", len(bones))

        vertices = vertices.reshape(vertices.size)
        primitives = [(GL_TRIANGLES, indices)]

        self.Shader = PositionShader(shader_path, vertices, primitives, textures)
        self.vertices = vertices.reshape((vertices.size // 3, 3))
        self.bones = bones
        self.vertex_bone_assignment = vertex_bone_assignment

    def updateTRSMatrices(self):
        t = glfw.get_time()
        t = t % (8 * np.pi)

        if (0 < t < 39 * np.pi / 20) or (  # No scaling we jumping
            121 * np.pi / 20 < t < 8 * np.pi
        ):
            self.S = pyrr.matrix44.create_from_scale([1.0, 1.0, 1.0])
            self.T = pyrr.matrix44.create_from_translation(
                [0, np.abs(np.sin(t + np.pi / 2)), 0]
            )
        elif 39 * np.pi / 20 < t < 41 * np.pi / 20:  # We are getting abducted
            matr1 = pyrr.matrix44.create_from_translation([0.0, 0.0, -1.0])
            matr2 = LA.inv(matr1)
            matr3 = pyrr.matrix44.create_from_scale(
                [
                    (np.sin(t * 10 + np.pi) + 1) / 2,
                    (np.sin(t * 10 + np.pi) + 1) / 2,
                    (np.sin(t * 10 + np.pi) + 1) / 2,
                ]
            )
            self.S = np.matmul(np.matmul(matr1, matr3), matr2)
        elif 119 * np.pi / 20 < t < 121 * np.pi / 20:  # We are getting de abducted
            matr1 = pyrr.matrix44.create_from_translation([0.0, 0.0, -1.0])
            matr2 = LA.inv(matr1)
            matr3 = pyrr.matrix44.create_from_scale(
                [
                    1 - (np.sin(t * 10 + np.pi) + 1) / 2,
                    1 - (np.sin(t * 10 + np.pi) + 1) / 2,
                    1 - (np.sin(t * 10 + np.pi) + 1) / 2,
                ]
            )
            self.S = np.matmul(np.matmul(matr1, matr3), matr2)
            # self.T = pyrr.matrix44.create_from_translation([0,3 * (np.sin(t) + 1), 0])
        else:  # We are currently in the spaceship
            self.S = pyrr.matrix44.create_from_scale([0.0, 0.0, 0.0])

    def updateVertices(self):
        # In this function, we first modify the bones' local matrices then we apply those to
        # vertices, based on distance between bones and vertices. For each bone, we calculate a
        # matrix that does D * A * B * C * B^-1 * A^-1.
        # D is the parent's matrix, if the bone has a parent. A is the translation matrix to the
        # bone head position. B is the rotation matrix to move into the correct coordinate system
        # (X in the bone's direction and the base vectors have the same norm as the bone). C is the
        # bone's local matrix.
        #
        # We then rely on NumPy to apply those matrices to every vertex in the mesh. We have
        # precalculated by which bone each vertex should be controlled, based on their initial
        # positions and the distances to each bone.

        t = glfw.get_time()

        # Reset the matrices. This is used to grab the bone's parent's matrix.
        for bone in self.bones.values():
            bone.matrix = None

        bone_matrices = []

        t1 = time.time()

        mat1 = np.array(pyrr.Matrix44.from_z_rotation(np.sin(t + np.pi / 2) / 2))
        self.bones["upper_arm.L"].local_matrix = mat1
        self.bones["upper_arm.R"].local_matrix = mat1
        self.bones["forearm.L"].local_matrix = mat1
        self.bones["forearm.R"].local_matrix = mat1

        mat2 = np.array(pyrr.Matrix44.from_z_rotation(-np.sin(t + np.pi / 2)))
        self.bones["thigh.L"].local_matrix = mat2
        self.bones["thigh.R"].local_matrix = mat2

        self.bones["spine.006"].local_matrix = np.array(
            pyrr.matrix44.create_from_axis_rotation([1.0, 0.0, 1.0], np.sin(t) / 2)
        )
        # bone["forearm.L"].local_matrix = np.array(pyrr.matrix44.create_from_scale([2 * np.abs(np.sin(t)), 1.0, 1.0]),dtype=np.float32,)

        for bone_name, bone in self.bones.items():

            # bone.local_matrix = pyrr.Matrix44.from_scale(np.array([2, 2., 2.]) * np.abs(np.sin(t)))
            # bone.local_matrix = pyrr.Matrix44.from_z_rotation(t/2).T
            # a = pyrr.matrix44.create_from_scale(np.array([np.abs(np.sin(t)), 1., 1.]))
            # b = pyrr.Matrix44.from_translation([0., 0., np.abs(np.sin(t))])
            # bone.local_matrix = np.matmul(a, b)
            # bone.local_matrix = a

            # Find the B matrix. It should move to the proper coordinate system. We look at the
            # scaling and two rotations required to get [1, 0, 0] to become (bone.tail - tail.head).
            # We then apply this transformation to [0, 1, 0] and [0, 0, 1].

            # To find the three axis values, we should find the rotation value needed on the Oz
            # plane to get in line with the bone. We then find the rotation needed to get to it
            # (in the plane: 0 - B - B projected onto the Oz plane) then we simply apply those two
            # rotations + the scaling onto all three unit axis (100, 010, 001) and it's won.

            parent_matrix = np.eye(4, dtype=np.float32)

            if bone.parent is not None:
                assert bone.parent.matrix is not None
                parent_matrix = bone.parent.matrix

            bone_head = np.matmul(parent_matrix, np.r_[bone.head, 1.0])[0:3]
            bone_tail = np.matmul(parent_matrix, np.r_[bone.tail, 1.0])[0:3]

            x = bone_tail - bone_head

            # find the angle between x and (x[0], x[1], 0)
            v = np.r_[x[0:2], 0.0]
            angle1 = np.arccos(np.dot(x, v) / (LA.norm(x) * LA.norm(v)))
            angle1 = -angle1 if x[2] >= 0 else angle1
            # find the angle between x and (1, 0, 0), on the Oz plane
            angle2 = np.arctan2(x[1], x[0])

            # Create the matrix that will move the two missing axis.
            mat1 = pyrr.Matrix33.from_scale([LA.norm(x)] * 3)
            mat2 = pyrr.matrix33.create_from_axis_rotation([0.0, 1.0, 0.0], angle1)
            mat3 = pyrr.matrix33.create_from_axis_rotation([0.0, 0.0, 1.0], angle2)
            mat = np.matmul(np.matmul(mat2, mat3), mat1)

            y = pyrr.matrix33.apply_to_vector(mat, [0.0, 1.0, 0.0])
            z = pyrr.matrix33.apply_to_vector(mat, [0.0, 0.0, 1.0])

            # We check that the calculation is right by applying the transformation matrix to
            # [1, 0, 0] and checking that we (almost) get bone.tail - tail.head.
            # x_bis = pyrr.matrix33.apply_to_vector(mat, [1.0, 0.0, 0.0])
            # np.testing.assert_almost_equal(x_bis, x)

            b = np.identity(4)
            b[0:3, 0:3] = np.c_[x, y, z]
            # b = b.T

            a = pyrr.matrix44.create_from_translation(bone_head).T

            bone.matrix = np.array(
                parent_matrix @ a @ b @ bone.local_matrix @ LA.inv(b) @ LA.inv(a),
                dtype=np.float32,
            )

            # Calculate the matrix.
            # bone.matrix = LA.multi_dot(
            #     [parent_matrix, a, b, bone.local_matrix, LA.inv(b), LA.inv(a)]
            # )
            bone_matrices.append(bone.matrix)

        t2 = time.time()

        bone_matrices = np.array(bone_matrices, dtype=np.float32)

        # Copy vertices.
        vertices = self.vertices.astype(np.float32)
        # Add a 1 at the end so that we can multiply them with the matrices.
        vertices = np.c_[vertices, np.repeat(np.float32(1.0), vertices.shape[0])]
        # Get a list of matrices to apply for each vertex. This is really
        # wasteful in memory but it offloads everything to NumPy.
        vertex_matrices = np.take(bone_matrices, self.vertex_bone_assignment, axis=0)
        # Calculate each matrix multiplication
        vertices = np.einsum("abc,ac->ab", vertex_matrices, vertices)[:, 0:3]

        t3 = time.time()

        print("a", int((t2 - t1) * 1000), "ms", "b", int((t3 - t2) * 1000), "ms")

        return vertices


class Bone:
    def __init__(self, name, parent, head, tail):
        self.name = name
        self.parent = parent
        self.head = head
        self.tail = tail
        self.local_matrix = homothety_matrix(1)

    def get_matrix(self):
        if self.parent is None:
            return this.local_matrix
        else:
            return this.local_matrix * self.parent.get_matrix()

    def center_point(self):
        # center = 0A + AB*.5
        return self.head + (self.tail - self.head) * 0.5


class WeightedVertex:
    def __init__(self, vertex, bone, weight):
        self.vertex = vertex
        self.bone = bone
        self.weight = weight


# This function returns the root bone of the given rig export.
def read_rig_export(path):
    found_root = False
    bones = {}

    # Each line is assumed to represent a bone.
    for line in open(path, encoding="UTF-8"):
        # Split each line into fields. There should be eight: name, parent's
        # name, head XYZ and tail XYZ.
        fields = line.strip().split("|")
        assert len(fields) == 8

        # Check we only find one bone that has no parent (i.e. is root).
        if fields[1] == "":
            assert not found_root  # We do not want two roots.
            found_root = True

        # Create the bone instance. In the parent property we put its parent name.
        head = np.array(
            [float(fields[2]), float(fields[3]), float(fields[4])], dtype=np.float32
        )
        tail = np.array(
            [float(fields[5]), float(fields[6]), float(fields[7])], dtype=np.float32
        )
        bones[fields[0]] = Bone(fields[0], fields[1], head, tail)

    root = None

    # We then loop through every bone found, looking for the root and replacing
    # parent names by references to bone instances.
    for bone in bones.values():
        if bone.parent == "":
            bone.parent = None
            root = bone
        else:
            bone.parent = bones[bone.parent]

    return bones


def read_rig_vertices_export(path):
    found_root = False
    vertices = {}

    # Each line is assumed to represent a weighted vertex.
    for line in open(path, encoding="UTF-8"):
        # Split each line into fields. There should be three: verticeindex, bonename and weight
        fields = line.strip().split("|")
        assert len(fields) == 3

        # Create the WeightedVertex instance.
        vertices[line] = WeightedVertex(fields[0], fields[1], fields[2])

    return vertices
