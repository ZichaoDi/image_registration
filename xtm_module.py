import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt
plt.ion()

def phantom3d(n=64, kind='modified shepp-logan'):
    def modified_shepp_logan():
        return np.array([
            [1, .6900, .920, .810, 0, 0, 0, 0, 0, 0],
            [-.8, .6624, .874, .780, 0, -.0184, 0, 0, 0, 0],
            [-.2, .1100, .310, .220, .22, 0, 0, -18, 0, 10],
            [-.2, .1600, .410, .280, -.22, 0, 0, 18, 0, 10],
            [.1, .2100, .250, .410, 0, .35, -.15, 0, 0, 0],
            [.1, .0460, .046, .050, 0, .1, .25, 0, 0, 0],
            [.1, .0460, .046, .050, 0, -.1, .25, 0, 0, 0],
            [.1, .0460, .023, .050, -.08, -.605, 0, 0, 0, 0],
            [.1, .0230, .023, .020, 0, -.606, 0, 0, 0, 0],
            [.1, .0230, .046, .020, .06, -.605, 0, 0, 0, 0]
        ])
    def shepp_logan():
        e = modified_shepp_logan()
        e[:, 0] = np.array([1, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        return e
    E = shepp_logan() if kind.lower().startswith('shepp-logan') and not kind.lower().startswith('modified') else modified_shepp_logan()
    rng = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    x, y, z = np.meshgrid(rng, rng, rng, indexing='ij')
    coords = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    p = np.zeros(coords.shape[1])
    for A, a, b, c, x0, y0, z0, phi, theta, psi in E:
        phi, theta, psi = np.deg2rad([phi, theta, psi])
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R = np.array([
            [cpsi * cphi - ctheta * sphi * spsi, cpsi * sphi + ctheta * cphi * spsi, spsi * stheta],
            [-spsi * cphi - ctheta * sphi * cpsi, -spsi * sphi + ctheta * cphi * cpsi, cpsi * stheta],
            [stheta * sphi, -stheta * cphi, ctheta]
        ])
        cp = R.dot(coords - np.array([[x0], [y0], [z0]]))
        mask = (cp[0]**2 / a**2 + cp[1]**2 / b**2 + cp[2]**2 / c**2) <= 1
        p[mask] += A
    return p.reshape(n, n, n)

def intersect_lines(line1, line2, tol=1e-14):
    L1 = np.atleast_2d(line1)
    L2 = np.atleast_2d(line2)
    if L1.shape[0] not in (1, L2.shape[0]) and L2.shape[0] not in (1, L1.shape[0]):
        raise ValueError
    N = max(L1.shape[0], L2.shape[0])
    L1 = np.repeat(L1, N // L1.shape[0], axis=0)
    L2 = np.repeat(L2, N // L2.shape[0], axis=0)
    denom = L1[:, 2] * L2[:, 3] - L2[:, 2] * L1[:, 3]
    par = np.abs(denom) < tol
    dx = L2[:, 0] - L1[:, 0]
    dy = L2[:, 1] - L1[:, 1]
    col = par & (np.abs(dx * L1[:, 3] - dy * L1[:, 2]) < tol)
    x0 = np.full(N, np.nan)
    y0 = np.full(N, np.nan)
    x0[col] = np.inf
    y0[col] = np.inf
    mask = ~par
    if np.any(mask):
        x1, y1, dx1, dy1 = L1[mask].T
        x2, y2, dx2, dy2 = L2[mask].T
        dn = denom[mask]
        dn_x = x2 - x1
        dn_y = y2 - y1
        x0_n = (x2 * dy2 * dx1 - dn_y * dx1 * dx2 - x1 * dy1 * dx2) / dn
        y0_n = (dn_x * dy1 * dy2 + y1 * dx1 * dy2 - y2 * dx2 * dy1) / dn
        x0[mask] = x0_n
        y0[mask] = y0_n
    return np.column_stack((x0, y0))

def line_position(points, lines, diag=False):
    P = np.atleast_2d(points)
    L = np.atleast_2d(lines)
    if diag and P.shape[0] == L.shape[0]:
        vx, vy = L[:, 2], L[:, 3]
        dx = P[:, 0] - L[:, 0]
        dy = P[:, 1] - L[:, 1]
        delta = vx * vx + vy * vy
        return (dx * vx + dy * vy) / np.where(delta == 0, 1, delta)
    vx, vy = L[:, 2], L[:, 3]
    dx = P[:, None, 0] - L[None, :, 0]
    dy = P[:, None, 1] - L[None, :, 1]
    delta = vx * vx + vy * vy
    return (dx * vx + dy * vy) / np.where(delta == 0, 1, delta)

def intersect_line_polygon(line, poly, tol=1e-14):
    poly = np.asarray(poly, float)
    edges = np.hstack((poly, np.roll(poly, -1, axis=0)))
    sup = np.column_stack((edges[:, 0], edges[:, 1], edges[:, 2] - edges[:, 0], edges[:, 3] - edges[:, 1]))
    pts = intersect_lines(line, sup, tol)
    inds = np.isfinite(pts[:, 0])
    pos = line_position(pts[inds], sup[inds], diag=True)
    mask = (pos > -tol) & (pos < 1 + tol)
    return np.unique(pts[inds][mask], axis=0)

def intersection_set(Source, Detector, xbox, ybox, theta, x, y, omega, m, dz, Tol):
    line = [Source[0], Source[1], Detector[0] - Source[0], Detector[1] - Source[1]]
    poly = list(zip(xbox, ybox))
    intersects = intersect_line_polygon(line, poly)
    if intersects.size == 0 or (np.unique(intersects[:, 0]).size == 1 and np.unique(intersects[:, 1]).size == 1):
        return [], np.array([]), np.array([])
    A = np.unique(intersects, axis=0)
    Ax, Ay = A[:, 0], A[:, 1]
    slope = (Ay[1] - Ay[0]) / (Ax[1] - Ax[0])
    intercept = (Ay[0] * Ax[1] - Ay[1] * Ax[0]) / (Ax[1] - Ax[0])
    Q1 = np.column_stack((x, slope * x + intercept))
    Q2 = np.column_stack(((y - intercept) / slope, y))
    Q = np.vstack((Q1, Q2))
    mask = (Q[:, 0] > xbox[0] - Tol) & (Q[:, 0] < xbox[2] + Tol) & (Q[:, 1] > ybox[0] - Tol) & (Q[:, 1] < ybox[1] + Tol)
    Q = np.unique(Q[mask], axis=0)
    diffs = np.diff(Q, axis=0)
    Lvec = np.hypot(diffs[:, 0], diffs[:, 1])
    QC = (Q[:-1] + Q[1:]) / 2
    idx = np.floor([(QC[:, 0] - omega[0]) / dz[0] + 1, (QC[:, 1] - omega[2]) / dz[1] + 1]).astype(int).T
    inside = (idx[:, 0] > 0) & (idx[:, 0] <= m[1]) & (idx[:, 1] > 0) & (idx[:, 1] <= m[0])
    idx = idx[inside]
    Lvec = Lvec[inside]
    linearInd = (idx[:, 1] - 1) * m[1] + (idx[:, 0] - 1)
    return idx.tolist(), Lvec, linearInd

def generate_noise(b1, b2, b3, GaussianSTD=0.01, s=1e7):
    b = np.vstack((b1, b2, b3)).T
    Smax = np.max(b[:, 2])
    bt = b.copy()
    btn = bt / Smax + np.random.normal(0, GaussianSTD, bt.shape)
    btn = np.clip(btn * Smax, 0, None)
    bn = np.zeros_like(b)
    for i in range(3):
        lam = b[:, i] / s
        bn[:, i] = np.random.poisson(lam) * s
    return btn, bn

def XTM_Tensor(nGrid=50, numTheta=30, Tol=1e-2):
    omega = np.array([-2, 2, -2, 2]) * Tol
    m = (nGrid, nGrid)
    dz = ((omega[1] - omega[0]) / m[1], (omega[3] - omega[2]) / m[0])
    alpha = np.arctan((omega[3] - omega[2]) / (omega[1] - omega[0]))
    Tau = omega[1] - omega[0]
    nTau = int(np.ceil(np.sqrt(2 * m[0] * m[1])))
    tol1 = 0.5 * nGrid
    detS0 = [Tau / 2 * np.tan(alpha) + tol1 * dz[0], -Tau / 2 - tol1 * dz[0]]
    detE0 = [Tau / 2 * np.tan(alpha) + tol1 * dz[0], Tau / 2 + tol1 * dz[0]]
    knot = np.linspace(detS0[1], detE0[1], nTau)
    DetKnot0 = np.column_stack((np.full(nTau, detS0[0]), knot))
    SourceS0 = [-Tau / 2 * np.tan(alpha) - tol1 * dz[0], -Tau / 2 - tol1 * dz[0]]
    SourceE0 = [-Tau / 2 * np.tan(alpha) - tol1 * dz[0], Tau / 2 + tol1 * dz[0]]
    knot = np.linspace(SourceS0[1], SourceE0[1], nTau)
    SourceKnot0 = np.column_stack((np.full(nTau, SourceS0[0]), knot))
    thetas = np.linspace(1, 360, numTheta)
    x = np.linspace(omega[0], omega[1], m[0] + 1)
    y = np.linspace(omega[2], omega[3], m[1] + 1)
    w = phantom3d(nGrid).sum(axis=2)
    XTM = np.zeros((numTheta, nTau))
    L = sparse.lil_matrix((numTheta * nTau, m[0] * m[1]))
    print(w.shape)

    for n, angle in enumerate(thetas):
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        DetKnot = DetKnot0.dot(R)
        SourceKnot = SourceKnot0.dot(R)
        Rdis = np.zeros(nTau)
        xbox = [omega[0], omega[0], omega[1], omega[1], omega[0]]
        ybox = [omega[2], omega[3], omega[3], omega[2], omega[2]]
        for i in range(nTau):
            idx, Lvec, lin = intersection_set(SourceKnot[i], DetKnot[i], xbox, ybox, theta, x, y, omega, m, dz, Tol)
            if Lvec.size and np.linalg.norm(Lvec) > 0:
                L[n * nTau + i, lin] = Lvec
                wmat = w * L[n * nTau + i].reshape(m)
                Rdis[i] = np.dot(np.ones(m[0]), np.dot(wmat, np.ones(m[1])))
        XTM[n, :] = Rdis
    return XTM, L

