"""Readable robot/hand materials with a retained, quieter starry stage."""


def style_hands(hands):
    for hand in hands:
        root = hand.mjcf_model
        shell = root.asset.add("material", name="piano_silver", rgba=(0.76, 0.80, 0.84, 1), specular=0.20, shininess=0.25)
        joint = root.asset.add("material", name="piano_joint", rgba=(0.24, 0.28, 0.32, 1), specular=0.10, shininess=0.15)
        for geom in root.find_all("geom"):
            if geom.mesh is not None:
                geom.material = joint if "knuckle" in geom.parent.name else shell
                geom.rgba = None


def style_stage(root):
    sky = root.find("texture", "skybox")
    if sky is not None:
        sky.markrgb = (0.58, 0.62, 0.68)
    floor = root.find("material", "grid")
    if floor is not None:
        floor.reflectance = 0.035
    for light in list(root.find_all("light")):
        light.remove()
    root.worldbody.add("light", name="piano_key_light", pos=(1.0, -1.4, 2.6), dir=(-0.8, 1.4, -1.5), directional=True, diffuse=(0.75, 0.72, 0.68), specular=(0.12, 0.12, 0.12))
    root.worldbody.add("light", name="piano_fill_light", pos=(-0.4, 1.2, 2.1), dir=(0.6, -1.2, -1.0), directional=True, castshadow=False, diffuse=(0.42, 0.46, 0.50), specular=(0.04, 0.04, 0.04))
    root.visual.headlight.ambient = (0.18, 0.18, 0.18)
    root.visual.headlight.diffuse = (0.10, 0.10, 0.10)
    root.visual.headlight.specular = (0.02, 0.02, 0.02)
