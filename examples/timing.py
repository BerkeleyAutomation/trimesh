import timeit
import numpy as np
import trimesh

def coo(mesh):
    edges = mesh.edges
    normals = mesh.face_normals 
   
    a = normals[:,0]
    b = normals[:,1]
    c = normals[:,2]
    d = np.einsum('ij,ij->i', normals, mesh.triangles[:,0,:])
    face_mats = np.array([a*a, a*b, a*c, a*d, b*b, b*c, b*d, c*c, c*d, d*d]).T
    vertex_quadrics = mesh.faces_sparse.dot(face_mats)
    #   mesh.faces_sparse.sum(axis=1)

def csr(mesh):
    edges = mesh.edges
    normals = mesh.face_normals 
   
    a = normals[:,0]
    b = normals[:,1]
    c = normals[:,2]
    d = np.einsum('ij,ij->i', normals, mesh.triangles[:,0,:])
    face_mats = np.array([a*a, a*b, a*c, a*d, b*b, b*c, b*d, c*c, c*d, d*d]).T
    vertex_quadrics = mesh.faces_sparse.tocsr().dot(face_mats)
    # mesh.faces_sparse.tocsr().sum(axis=1)

mesh = trimesh.load('/home/mjd3/working/catkin_ws/src/trimesh/models/cycloidal.ply')
# print('COO time, 10000 loops: {:.6f} s per loop'.format(
#       timeit.timeit('coo(mesh)', 
#       setup="from __main__ import mesh, coo", number=10000)/10000))

# print('CSR time, 10000 loops: {:.6f} s per loop'.format(
#       timeit.timeit('csr(mesh)', 
#       setup="from __main__ import mesh, csr", number=10000)/10000))


# r = np.random.randint(0, 100, 50)
# x=trimesh.grouping.unique_bincount(r, return_counts=False)
# print(x)

from scipy.sparse import coo_matrix
from trimesh.geometry import vertex_face_indices
mesh1 = trimesh.load('models/cycloidal.ply')
mesh2 = trimesh.load('models/bunny.ply')

print(vertex_face_indices(len(mesh1.vertices), mesh1.faces, sparse=None))

# def vertex_faces2(x):
#     return [np.nonzero(i)[1] for i in x.faces_sparse.todense()]

# print('vertex_faces1 time, 1000 loops: {:.6f} s per loop'.format(
#       timeit.timeit('vertex_faces1(mesh2)', 
#       setup="from __main__ import mesh2, vertex_faces1", number=1000)/1000))
# print('vertex_faces2 time, 10 loops: {:.6f} s per loop'.format(
#       timeit.timeit('vertex_faces2(mesh1)', 
#       setup="from __main__ import mesh1, vertex_faces2", number=10)/10))

# print('vertex_faces1 time, 1000 loops: {:.6f} s per loop'.format(
#       timeit.timeit('vertex_faces1(mesh2)', 
#       setup="from __main__ import mesh2, vertex_faces1", number=1000)/1000))
# print('vertex_faces2 time, 10 loops: {:.6f} s per loop'.format(
#       timeit.timeit('vertex_faces2(mesh2)', 
#       setup="from __main__ import mesh2, vertex_faces2", number=10)/10))
