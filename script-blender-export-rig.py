import bpy

print('### RIG EXPORT START ###')

rig = bpy.data.objects['RigToImportV2']
mesh = bpy.data.objects['AlienToImport']

mesh_verts = mesh.data.vertices
mesh_group_names = [g.name for g in mesh.vertex_groups]

stringvertices = ''

# Loop on each bone.
for b in rig.pose.bones:
    string = (b.name + '|' +                                              # name
        (b.parent.name if b.parent is not None else '') + '|' +           # parent's name
        str(b.head.x) + '|' + str(b.head.y) + '|' + str(b.head.z) + '|' + # head coordinates
        str(b.tail.x) + '|' + str(b.tail.y) + '|' + str(b.tail.z))        # tail coordinates

    # Make sure we don't have anything that would break the formatting.
    assert(string.count('|') == 7)
    assert(string.count('\n') == 0)
    print(string)
    
    #Search the vertices weighted by this bone
    if b.name not in mesh_group_names:
        continue

    gidx = mesh.vertex_groups[b.name].index

    bone_verts = [v for v in mesh_verts if gidx in [g.group for g in v.groups]]

    for v in bone_verts:
        for g in v.groups:
            if g.group == gidx: 
                w = g.weight
                stringvertices = (stringvertices + str(v.index) + '|' + b.name + '|' + str(w) + '\n')
    
print('### RIG EXPORT END ###')
    
print(stringvertices)

print('### RIGVERTICES EXPORT END ###')
