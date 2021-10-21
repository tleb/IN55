from opengl_fcts import *
import time
import numpy as np
import numpy.linalg as LA
import stl
import rig
import sys
import meshes


def main():
    window = Window(1024, 768, "SpaceAnimation")

    if not window.Window:
        return

    window.initViewMatrix(eye=[0.0, 0.0, 5.0])

    rc = rig.RiggedMesh(
        "../alien.stl",
        "../rig-export.txt",
        "./alien",
        [
            ("./texture-alien.jpg", GL_TEXTURE1, "texID1"),
        ],
    )
    rc.R = np.matmul(
        pyrr.matrix44.create_from_axis_rotation([0.0, 1.0, 0.0], -np.pi / 2),
        pyrr.matrix44.create_from_axis_rotation([0.0, 0.0, 1.0], -np.pi / 2),
    )

    sc = meshes.Soucoupe()
    sc.rotate([1.0, 0.0, 0.0], -np.pi / 2)

    ast = meshes.Asteroide()

    plan = meshes.Planete()
    plan.translate((0.0, -5.0, 0.0))

    objects = [rc, sc, ast, plan]

    window.render(objects)


if __name__ == "__main__":
    main()
