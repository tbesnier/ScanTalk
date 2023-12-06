def compute_operators(verts, faces, k_eig, normals=None):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.

    See get_operators() for a similar routine that wraps this one with a layer of caching.

    Torch in / torch out.

    Arguments:
      - vertices: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use

    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian 
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient

    PyTorch doesn't seem to like complex sparse matrices, so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately, rather than as one complex sparse matrix.

    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix, like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """

    device = verts.device
    dtype = verts.dtype
    V = verts.shape[0]
    is_cloud = faces.numel() == 0

    eps = 1e-8

    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces)
    frames = build_tangent_frames(verts, faces, normals=normals)
    frames_np = toNP(frames)

    # Build the scalar Laplacian
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
        massvec_np = M.diagonal()
    else:
        # L, M = robust_laplacian.mesh_laplacian(verts_np, faces_np)
        # massvec_np = M.diagonal()
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts_np, faces_np)
        massvec_np += eps * np.mean(massvec_np)
    
    if(np.isnan(L.data).any()):
        raise RuntimeError("NaN Laplace matrix")
    if(np.isnan(massvec_np).any()):
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # === Compute the eigenbasis
    if k_eig > 0:

        # Prepare matrices
        L_eigsh = (L + scipy.sparse.identity(L.shape[0])*eps).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise, but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma)
            
                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

                break
            except Exception as e:
                print(e)
                if(failcount > 3):
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10**failcount)


    else: #k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0],0))


    # == Build gradient matrices

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood
    if is_cloud:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts, edges, edge_vecs)


    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)
    
    # === Convert back to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = utils.sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = utils.sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    return frames, massvec, L, evals, evecs, gradX, gradY
