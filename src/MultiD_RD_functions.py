import os
import time
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree

"""
This code is modified code from 10.5281/zenodo.4625675, 
please cite this software if used in a paper.
"""


def paralell_hist_loop(
    radii_and_indices, kdtree, particles, mins, maxs, N_radii, dr, eps, rho
):
    """RDF histogram loop process for multiprocessing"""
    N, d = particles.shape
    g_r_partial = np.zeros(shape=(N_radii))

    for r_idx, r in radii_and_indices:
        r_idx = int(r_idx)
        # find all particles that are at least r + dr away from the edges of the box
        valid_idxs = np.bitwise_and.reduce(
            [
                (particles[:, i] - (r + dr) >= mins[i])
                & (particles[:, i] + (r + dr) <= maxs[i])
                for i in range(d)
            ]
        )
        valid_particles = particles[valid_idxs]

        # compute n_i(r) for valid particles.
        for particle in valid_particles:
            n = kdtree.query_ball_point(
                particle, r + dr - eps, return_length=True
            ) - kdtree.query_ball_point(particle, r, return_length=True)
            g_r_partial[r_idx] += n

        # normalize
        n_valid = len(valid_particles)
        shell_vol = (
            (4 / 3) * np.pi * ((r + dr) ** 3 - r**3)
            if d == 3
            else np.pi * ((r + dr) ** 2 - r**2)
        )
        g_r_partial[r_idx] /= n_valid * shell_vol * rho

    return g_r_partial


def multid_rdf(
    particles_1,
    particles_2,
    r_max,
    dr,
    boxdims=([[0.0, 0.0], [132.0, 132.0]]),
    rho=None,
    eps=1e-15,
    parallel=True,
    progress=False,
):
    """
    Computes 2D or 3D radial distribution function g(r) of a set of particle
    coordinates of shape (N, d). Particle must be placed in a 2D or 3D cuboidal
    box of dimensions [width x height (x depth)].

    Parameters
    ----------
    particles_1 : (N, d) np.array
        Set of particles from which to compute the radial distribution function
        g(r) with respect to particles_2.
        Must be of shape (N, 2) or (N, 3) for 2D and 3D coordinates
        repsectively.
    particles_2 : (N, d) np.array
        Set of particles from which the rdf will be computed with respect to
        particles_1.
        Must be of shape (N, 2) or (N, 3) for 2D and 3D coordinates
        repsectively.
    r_max : float
        Max radius over which g(r) is computed.
    dr : float
        dr over which g(r) steps are computed
    boxdims : (n, 2) np.array
        Dimensions of box, default ([[0., 132.], [0., 132.]])
    rho : float, optional
        Number density. If left as None, box dimensions will be inferred from
        the particles and the number density will be calculated accordingly.
    eps : float, optional
        Epsilon value used to find particles less than or equal to a distance
        in KDTree.
    parallel : bool, optional
        Option to enable or disable multiprocessing. Enabling this affords
        significant increases in speed.
    progress : bool, optional
        Set to False to disable progress readout (only valid when
        parallel=False).


    Returns
    -------
    g_r : (n_radii) np.array
        radial distribution function values g(r).
    radii : (n_radii) np.array
        radii over which g(r) is computed
    """

    if not isinstance(particles_1, np.ndarray):
        particles_1 = np.array(particles_1)
    if not isinstance(particles_2, np.ndarray):
        particles_2 = np.array(particles_2)
    # assert particles array is correct shape
    shape_err_msg = "particles should be an array of shape N x d, where N is \
                     the number of particles and d is the number of dimensions."
    assert len(particles_1.shape) == 2, shape_err_msg
    # assert particle coords are 2 or 3 dimensional
    assert particles_1.shape[-1] in [
        2,
        3,
    ], "RDF can only be computed in 2 or 3 \
                                           dimensions."
    assert len(particles_2.shape) == 2, shape_err_msg
    # assert particle coords are 2 or 3 dimensional
    assert particles_2.shape[-1] in [
        2,
        3,
    ], "RDF can only be computed in 2 or 3 \
                                           dimensions."
    start = time.time()

    radii = np.arange(dr, r_max, dr)

    N_particles1, d = particles_1.shape
    N_particles2, d = particles_2.shape

    mins = np.min(boxdims, axis=0)
    maxs = np.max(boxdims, axis=0)

    particles_2 = particles_2 - mins
    particles_1 = particles_1 - mins

    if not rho:
        rho = N_particles2 / np.prod(maxs)  # number density

    # create a KDTree for fast nearest-neighbor lookup of particles
    tree = cKDTree(particles_2)

    if parallel:
        N_radii = len(radii)
        radii_and_indices = np.stack([np.arange(N_radii), radii], axis=1)
        radii_splits = np.array_split(radii_and_indices, os.cpu_count(), axis=0)
        values = [
            (radii_splits[i], tree, particles_1, mins, maxs, N_radii, dr, eps, rho)
            for i in range(len(radii_splits))
        ]
        with Pool() as pool:
            results = pool.starmap(paralell_hist_loop, values)
        g_r = np.sum(results, axis=0)
    else:
        g_r = np.zeros(shape=(len(radii)))
        for r_idx, r in enumerate(radii):
            valid_idxs = np.bitwise_and.reduce(
                [
                    (particles_1[:, i] - (r + dr) >= mins[i])
                    & (particles_1[:, i] + (r + dr) <= maxs[i])
                    for i in range(d)
                ]
            )
            valid_particles = particles_1[valid_idxs]

            # compute n_i(r) for valid particles.
            for particle in valid_particles:
                n = tree.query_ball_point(
                    particle, r + dr - eps, return_length=True
                ) - tree.query_ball_point(particle, r, return_length=True)
                g_r[r_idx] += n

            # normalize
            n_valid = len(valid_particles)
            shell_vol = (
                (4 / 3) * np.pi * ((r + dr) ** 3 - r**3)
                if d == 3
                else np.pi * ((r + dr) ** 2 - r**2)
            )
            g_r[r_idx] /= n_valid * shell_vol * rho

            if progress:
                print(
                    "Computing RDF     Radius {}/{}    Time elapsed: {:.3f} s".format(
                        r_idx + 1, len(radii), time.time() - start
                    ),
                    end="\r",
                    flush=True,
                )

    return g_r, radii
