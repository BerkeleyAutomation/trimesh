"""
remesh.py
-------------

Deal with re- triangulation of existing meshes.
"""

import numpy as np
import time

from .constants import log
from . import geometry
from . import grouping
from . import triangles
from . import util


def subdivide(vertices,
              faces,
              face_index=None):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those faces will
    be subdivided and their neighbors won't be modified making
    the mesh no longer "watertight."

    Parameters
    ----------
    vertices : (n, 3) float
      Vertices in space
    faces : (n, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces

    Returns
    ----------
    new_vertices : (n, 3) float
      Vertices in space
    new_faces : (n, 3) int
      Remeshed faces
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c,3) int set of vertex indices
    faces = faces[face_index]
    # the (c, 3, 3) float set of points in the triangles
    triangles = vertices[faces]
    # the 3 midpoints of each triangle edge
    # stacked to a (3 * c, 3) float
    mid = np.vstack([triangles[:, g, :].mean(axis=1)
                     for g in [[0, 1],
                               [1, 2],
                               [2, 0]]])

    # for adjacent faces we are going to be generating
    # the same midpoint twice so merge them here
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    unique, inverse = grouping.unique_rows(mid)
    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(vertices)

    # the new faces with correct winding
    f = np.column_stack([faces[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    return new_vertices, new_faces


def subdivide_to_size(vertices,
                      faces,
                      max_edge,
                      max_iter=10):
    """
    Subdivide a mesh until every edge is shorter than a
    specified length.

    Will return a triangle soup, not a nicely structured mesh.

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indices of vertices which make up triangles
    max_edge : float
      Maximum length of any edge in the result
    max_iter : int
      The maximum number of times to run subdivision

    Returns
    ------------
    vertices : (j, 3) float
      Vertices in space
    faces : (q, 3) int
      Indices of vertices
    """
    # store completed
    done_face = []
    done_vert = []

    # copy inputs and make sure dtype is correct
    current_faces = np.array(faces,
                             dtype=np.int64,
                             copy=True)
    current_vertices = np.array(vertices,
                                dtype=np.float64,
                                copy=True)

    # loop through iteration cap
    for i in range(max_iter + 1):
        # (n, 3, 3) float triangle soup
        triangles = current_vertices[current_faces]

        # compute the length of every triangle edge
        edge_lengths = (np.diff(triangles[:, [0, 1, 2, 0]],
                                axis=1) ** 2).sum(axis=2) ** .5
        too_long = (edge_lengths > max_edge).any(axis=1)

        # clean up the faces a little bit so we don't
        # store a ton of unused vertices
        unique, inverse = np.unique(
            current_faces[np.logical_not(too_long)],
            return_inverse=True)

        # store vertices and faces meeting criteria
        done_vert.append(current_vertices[unique])
        done_face.append(inverse.reshape((-1, 3)))

        # met our goals so abort
        if not too_long.any():
            break

        # run subdivision again
        (current_vertices,
         current_faces) = subdivide(current_vertices,
                                    current_faces[too_long])

    # stack sequence into nice (n, 3) arrays
    vertices, faces = util.append_faces(done_vert,
                                        done_face)

    return vertices, faces

def simplify(mesh, target_count, aggressiveness=7, max_iter=100):

    if len(mesh.faces) <= target_count:
        return mesh
    
    faces = np.copy(mesh.faces)
    vertices = np.copy(mesh.vertices)
    
    # First create quadrics for each triangle and calculate vertex quadrics from these
    tic = time.time()
    a = mesh.face_normals[:,0]
    b = mesh.face_normals[:,1]
    c = mesh.face_normals[:,2]
    d = np.einsum('ij,ij->i', mesh.face_normals, mesh.triangles[:,0,:])
    face_quadrics = np.array([a*a, a*b, a*c, a*d, b*b, b*c, b*d, c*c, c*d, d*d]).T
    vertex_quadrics = mesh.faces_sparse.dot(face_quadrics)
    log.debug('%s found in %.6f', "vertex_quadrics", time.time() - tic)
    
    # Next, find initial edge errors
    tic = time.time()
    edge_errors, optimal_vertices = _calculate_edge_errors(faces, vertices, mesh.edges, vertex_quadrics) 
    edge_errors = edge_errors.reshape((len(faces), 3))
    optimal_vertices = optimal_vertices.reshape((len(faces), 3, 3))
    log.debug('%s found in %.6f', "edge_errors", time.time() - tic)

    # If mesh is not watertight, find the boundary vertices (boundary edge vertices)
    tic = time.time()
    boundary_vertices = np.zeros(len(vertices))
    if not mesh.is_watertight:
        boundary_vert_inds = _find_boundary_vertices(mesh.edges_unique, mesh.face_adjacency_edges)
        boundary_vertices[boundary_vert_inds] = 1
    log.debug('%s found in %.6f', "boundary_vertices", time.time() - tic)

    # Main loop: Iteratively remove faces
    for i in range(max_iter):
        print("Iteration: {}, Triangles: {}".format(i, len(faces)))
        if (len(faces) <= target_count):
            break
        
        # TODO: FIX THIS LINE. Probably want to update the face array in place without dumping all of the caches?
        # mesh.update_faces(face_mask)
        
        refs = _find_mesh_refs(mesh)
        dirty_faces = np.zeros(len(faces), dtype=np.bool_)
        
        # threshold = (1e-9)*(i+3)**aggressiveness
        threshold = np.finfo(float).eps

        # Find all edges with error below threshold
        candidate_inds = np.where(edge_errors < threshold)
        candidate_vert_inds = faces[candidate_inds]
        candidate_rep_verts = optimal_vertices[candidate_inds]

        # Filter edges that have vertices on the boundary
        candidate_edges = np.stack((faces[candidate_inds],
                                    faces[candidate_inds[0], (candidate_inds[1]+1)%3]))
        boundary_mask = boundary_vertices[candidate_edges]
        boundary_mask = np.logical_or(boundary_mask[0], boundary_mask[1])
        candidate_vert_inds = candidate_vert_inds[~boundary_mask]
        candidate_rep_verts = candidate_rep_verts[~boundary_mask]
        candidate_inds = (candidate_inds[0][~boundary_mask], candidate_inds[1][~boundary_mask])

        # TODO: Check whether replacing the vertex will result in the normal of the triangle flipping
        # pre_normals = triangles.cross(vertices[faces])
        # post_faces_sparse = geometry.index_sparse(len(vertices), faces).toarray().astype(np.int_)
        # vert_rep_row_inds = faces[candidate_inds[0], (candidate_inds[1]+1) % 3]
        # post_faces_sparse[vert_rep_row_inds,:] = post_faces_sparse[candidate_vert_inds,:]
        # import pdb; pdb.set_trace()
        # post_faces = np.where(post_faces_sparse == 1)
        # post_verts = np.copy(vertices)
        # post_verts[candidate_vert_inds] = candidate_rep_verts
        # post_normals = triangles.cross(post_verts[post_faces])
        # normal_dots = np.einsum('ij,ij->i', pre_normals, post_normals)
        # flip_post_verts = faces[np.where(normal_dots < 0.2)]

        # import pdb; pdb.set_trace()

        # candidate_vert_inds = grouping.unique_bincount(candidate_edges.flatten(), 
        #                                                minlength=np.max(candidate_edges), 
        #                                                return_inverse=False)
        # candidate_verts = mesh.vertices[candidate_vert_inds]
        # candidate_vertex_quadrics = vertex_quadrics[candidate_vert_inds,:]
        # p_results = _calculate_edge_errors(candidate_faces, candidate_verts, candidate_edges.T, candidate_vertex_quadrics)[1]

def _find_mesh_refs(mesh):

    # Init reference ID list: get face index and 
    # index within face array for each vertex
    vids, fids = mesh.faces_sparse.nonzero()
    vfids = np.tile(np.arange(3), len(mesh.faces))
    refs = np.stack((vids, fids, vfids))
    return refs

def _find_boundary_vertices(edges_unique, face_adjacency_edges):

    # TODO: Determine whether this is faster than group_rows!
    # Map unique edges and face adjacency edges to 1D
    unique_edge_hashes = grouping.hashable_rows(edges_unique)
    face_adjacency_edge_hashes = grouping.hashable_rows(face_adjacency_edges)
    
    # Find edges that exist in unique edges, but not in face adjacency edges
    boundary_edge_inds = np.searchsorted(unique_edge_hashes, 
                                         np.setdiff1d(unique_edge_hashes, 
                                                      face_adjacency_edge_hashes, 
                                                      assume_unique=True))
    boundary_vert_inds = edges_unique[boundary_edge_inds].reshape(-1)
    return boundary_vert_inds

def _calculate_edge_errors(faces, verts, edges, vert_quadrics):

    # Create array for edge errors and resulting points
    edge_errors = np.zeros(edges.shape[0])
    p_results = np.zeros((edges.shape[0], verts.shape[1]))

    # Create edge quadrics from vertex quadrics
    edges_sparse = geometry.index_sparse(verts.shape[0], edges).T
    edge_quadrics = edges_sparse.dot(vert_quadrics)

    # Find determinants of quadric matrices; if nonzero then proceeed with error calculation
    edge_quadric_dets = _calculate_determinants(edge_quadrics, (0,1,2,1,4,5,2,5,7))
    zero_det_mask = edge_quadric_dets == 0

    # Calculation for nonzero determinants
    nonzero_edge_quadrics = edge_quadrics[~zero_det_mask,:]
    nonzero_edge_dets = edge_quadric_dets[~zero_det_mask]
    vx_dets = _calculate_determinants(nonzero_edge_quadrics, (1,2,3,4,5,6,5,7,8))
    vy_dets = _calculate_determinants(nonzero_edge_quadrics, (0,2,3,1,5,6,2,7,8))
    vz_dets = _calculate_determinants(nonzero_edge_quadrics, (0,1,3,1,4,6,2,5,8))
    px = -vx_dets/nonzero_edge_dets
    py = vy_dets/nonzero_edge_dets
    pz = -vz_dets/nonzero_edge_dets
    edge_errors[~zero_det_mask] = _vertex_errors(nonzero_edge_quadrics, px, py, pz)
    p_results[~zero_det_mask,:] = np.stack((px,py,pz), axis=1) 

    # Calculation for zero determinants - fallback to midpoint and vertices as possible points and take min error
    zero_edge_verts = verts[edges[zero_det_mask,:]]
    zero_edge_midpoints = np.sum(zero_edge_verts, axis=1) / 2
    zero_edge_points = np.concatenate((zero_edge_verts, zero_edge_midpoints[:,None,:]), axis=1)
    zero_edge_errs = np.stack((_vertex_errors(edge_quadrics[zero_det_mask], *zero_edge_points[:,0,:].T),
                               _vertex_errors(edge_quadrics[zero_det_mask], *zero_edge_points[:,1,:].T),
                               _vertex_errors(edge_quadrics[zero_det_mask], *zero_edge_points[:,2,:].T)))
    edge_errors[zero_det_mask] = np.min(zero_edge_errs, axis=0)
    p_results[zero_det_mask,:] = zero_edge_points[range(len(zero_edge_points)),np.argmin(zero_edge_errs, axis=0),:]

    return edge_errors, p_results

# Error between quadric and vertex
def _vertex_errors(q, x, y, z):
    return (q[...,0]*x*x + 2*q[...,1]*x*y + 2*q[...,2]*x*z + 2*q[...,3]*x + q[...,4]*y*y
            + 2*q[...,5]*y*z + 2*q[...,6]*y + q[...,7]*z*z + 2*q[...,8]*z + q[...,9])

# Determinant for symmetric matrix, specified as array (10,)
def _calculate_determinants(mats, inds):
    return (mats[...,inds[0]]*mats[...,inds[4]]*mats[...,inds[8]] + mats[...,inds[2]]*mats[...,inds[3]]*mats[...,inds[7]]
            + mats[...,inds[1]]*mats[...,inds[5]]*mats[...,inds[6]] - mats[...,inds[2]]*mats[...,inds[4]]*mats[...,inds[6]]
            - mats[...,inds[0]]*mats[...,inds[5]]*mats[...,inds[7]] - mats[...,inds[1]]*mats[...,inds[3]]*mats[...,inds[8]])
