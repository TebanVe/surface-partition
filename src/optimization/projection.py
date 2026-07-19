import time
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from scipy.optimize import minimize

from ..logging_config import get_logger

def orthogonal_projection_iterative(A: np.ndarray, c: np.ndarray, d: np.ndarray, v: np.ndarray,
                                  max_iter: int = 1000, tol: float = 1e-10,
                                  logger: Optional[logging.Logger] = None,
                                  _prof=None) -> np.ndarray:
    """
    Implements the orthogonal projection algorithm for partition and area constraints
    as described in the paper "Partitions of Minimal Length on Manifolds".
    This is the iterative version that ensures better convergence.
    
    The algorithm projects a matrix A onto the intersection of two constraint sets:
    1. Partition constraint: Each row sums to 1 (Σᵢ Aᵢⱼ = 1 for all i)
    2. Area constraint: Weighted column sums equal target areas (v^T A = d)
    
    Args:
        A: Matrix of size N x n containing the density functions
        c: Vector of size n containing the target column sums (usually ones)
        d: Vector of size n containing the target area constraints
        v: Vector of size N containing the sum of mass matrix columns (v = 1ᵀM)
        max_iter: Maximum number of iterations for the alternating projection
        tol: Tolerance for convergence
        logger: Logger instance for progress tracking
        
    Returns:
        The orthogonally projected matrix A that satisfies the constraints
        
    Raises:
        ValueError: If input dimensions are incompatible
        RuntimeError: If projection fails to converge
    """
    if logger is None:
        logger = get_logger(__name__)

    if _prof is not None:
        t0 = time.perf_counter()
        _prof.add_counter('projection_invocations', 1)

    # Validate input dimensions
    N, n = A.shape
    if len(c) != n or len(d) != n or len(v) != N:
        raise ValueError(f"Dimension mismatch: A({N}x{n}), c({len(c)}), d({len(d)}), v({len(v)})")

    logger.debug(f"Starting orthogonal projection: {N}x{n} matrix, max_iter={max_iter}, tol={tol}")
    
    A = A.copy()  # Make a copy to avoid modifying the input
    
    # Initial normalization to satisfy partition constraint
    row_sums = np.sum(A, axis=1)
    mask = row_sums > 0  # Avoid division by zero
    A[mask] = A[mask] / row_sums[mask, np.newaxis]
    A[~mask] = 1.0/n  # Set uniform distribution for zero rows
    
    # Small regularization to avoid numerical issues
    epsilon = 1e-10

    # The coupling matrix C = ||v||^2 (I - J/n) and its epsilon-regularized form
    # depend only on v and n, which are fixed for the whole call; build once
    # (Change B / audit #4). C_reg is the matrix actually solved each iteration.
    v_norm_squared = np.sum(v**2)
    C = np.full((n, n), -v_norm_squared/n)
    np.fill_diagonal(C, v_norm_squared - v_norm_squared/n)
    C_reg = C + epsilon * np.eye(n)
    C_reg_reduced = C[:-1, :-1] + epsilon * np.eye(n-1)  # singular-fallback path

    # Scalar-residual stall tracker (replaces the O(V*N) A_prev copy + allclose).
    # Break only after the residual fails to improve on its best value for several
    # CONSECUTIVE iterations, so a single non-monotone tick-up of the alternating
    # projection is not mistaken for convergence (Change B / audit #4).
    best_max_error = np.inf
    stall_count = 0
    stall_patience = 5

    for iter in range(max_iter):
        # Step 1: Calculate line sum error (N x 1 column vector)
        e = np.sum(A, axis=1) - np.ones(N)  # Each row should sum to 1

        # Step 2: Calculate column scalar product error (n x 1 column vector)
        f = v @ A - d

        # Step 4: Calculate q vector
        q = f - np.dot(v, e)/n

        # Step 5: Solve for lambda (against the precomputed regularized C)
        try:
            # Try solving the full system first
            lambda_vec = np.linalg.solve(C_reg, q)
        except np.linalg.LinAlgError:
            # Fall back to reduced system if full system fails
            logger.warning(f"Iteration {iter}: Full system singular, using reduced system")
            lambda_vec = np.zeros(n)
            lambda_vec[:-1] = np.linalg.solve(C_reg_reduced, q[:-1])
        
        # Step 6: Calculate S
        S = np.sum(lambda_vec)
        
        # Step 7: Calculate eta vector
        eta = (e - S * v)/n
        
        # Step 8: Calculate orthogonal correction
        A_orth = np.outer(eta, np.ones(n)) + np.outer(v, lambda_vec)
        
        # Step 9: Apply correction
        A = A - A_orth
        
        # Step 10: Ensure non-negativity
        A = np.maximum(A, 0)
        
        # Step 11: Normalize rows to ensure partition constraint
        row_sums = np.sum(A, axis=1)
        mask = row_sums > epsilon  # Avoid division by zero
        A[mask] = A[mask] / row_sums[mask, np.newaxis]
        A[~mask] = 1.0/n  # Set uniform distribution for zero rows
        
        # Step 12: Project onto area constraints
        area_sums = v @ A
        scale_factors = d / (area_sums + epsilon)  # Add epsilon to avoid division by zero
        A = A * scale_factors[np.newaxis, :]
        
        # Check convergence of both constraints
        row_sum_error = np.max(np.abs(np.sum(A, axis=1) - 1))
        area_error = np.max(np.abs(v @ A - d))
        max_error = max(row_sum_error, area_error)

        # Log progress every 10 iterations
        if iter % 10 == 0:
            logger.debug(f"Iteration {iter}: row_error={row_sum_error:.2e}, area_error={area_error:.2e}")

        if row_sum_error < tol and area_error < tol:
            logger.info(f"Projection converged after {iter+1} iterations")
            logger.info(f"Final errors: row={row_sum_error:.2e}, area={area_error:.2e}")
            break

        # Scalar-residual stall test (replaces the O(V*N) allclose on A vs A_prev).
        # Count CONSECUTIVE iterations that fail to improve the best residual by at
        # least 1e-2*tol; break on a sustained plateau. Unlike a single-step delta,
        # this tolerates the alternating projection's non-monotone ticks. The test is
        # magnitude-agnostic: a genuine stall ABOVE tolerance also breaks here (rather
        # than running to max_iter) and is then caught by the feasibility validation
        # below, which raises if the result is worse than 10*tol.
        if max_error < best_max_error - 1e-2 * tol:
            best_max_error = max_error
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= stall_patience:
                if max_error > tol:
                    logger.warning(
                        f"Projection stalled after {iter+1} iterations above tol "
                        f"(max_error={max_error:.2e} > tol={tol:.1e}); "
                        f"row={row_sum_error:.2e}, area={area_error:.2e}"
                    )
                else:
                    logger.debug(
                        f"Projection stalled after {iter+1} iterations near tol: "
                        f"row={row_sum_error:.2e}, area={area_error:.2e}"
                    )
                break
    else:
        # Loop completed without convergence
        logger.warning(f"Projection did not converge after {max_iter} iterations")
        logger.warning(f"Final errors: row={row_sum_error:.2e}, area={area_error:.2e}")

    if _prof is not None:
        # `iter` survives the for/else: it holds the iteration we broke on,
        # or max_iter-1 if the loop ran to completion.
        _prof.add_counter('projection_inner_iters_total', iter + 1)
        _prof.record('projection', time.perf_counter() - t0)

    # Validate final result
    final_row_error = np.max(np.abs(np.sum(A, axis=1) - 1))
    final_area_error = np.max(np.abs(v @ A - d))
    
    if final_row_error > 10*tol or final_area_error > 10*tol:
        logger.error(f"Projection failed: row_error={final_row_error:.2e}, area_error={final_area_error:.2e}")
        raise RuntimeError("Orthogonal projection failed to achieve required tolerance")
    
    return A

def orthogonal_projection_direct(A: np.ndarray, c: np.ndarray, d: np.ndarray, v: np.ndarray,
                               logger: Optional[logging.Logger] = None,
                               _prof=None) -> np.ndarray:
    """
    Implements the orthogonal projection algorithm for partition and area constraints
    as described in the paper "Partitions of Minimal Length on Manifolds".
    This is the direct, non-iterative version.
    
    Args:
        A: Matrix of size N x n containing the density functions
        c: Vector of size n containing the target column sums (usually ones)
        d: Vector of size n containing the target area constraints
        v: Vector of size N containing the sum of mass matrix columns (v = 1ᵀM)
        logger: Logger instance for progress tracking
        
    Returns:
        The orthogonally projected matrix A that satisfies the constraints
        
    Raises:
        ValueError: If input dimensions are incompatible
        RuntimeError: If projection fails
    """
    if logger is None:
        logger = get_logger(__name__)

    if _prof is not None:
        t0 = time.perf_counter()
        _prof.add_counter('projection_invocations', 1)

    # Validate input dimensions
    N, n = A.shape
    if len(c) != n or len(d) != n or len(v) != N:
        raise ValueError(f"Dimension mismatch: A({N}x{n}), c({len(c)}), d({len(d)}), v({len(v)})")

    logger.debug(f"Starting direct orthogonal projection: {N}x{n} matrix")
    
    A = A.copy()  # Make a copy to avoid modifying the input
    
    # Initial normalization to satisfy partition constraint
    row_sums = np.sum(A, axis=1)
    mask = row_sums > 0  # Avoid division by zero
    A[mask] = A[mask] / row_sums[mask, np.newaxis]
    A[~mask] = 1.0/n  # Set uniform distribution for zero rows
    
    # Small regularization to avoid numerical issues
    epsilon = 1e-10
    
    # Step 1: Calculate line sum error (N x 1 column vector)
    e = np.sum(A, axis=1) - np.ones(N)  # Each row should sum to 1
    
    # Step 2: Calculate column scalar product error (n x 1 column vector)
    f = v @ A - d
    
    # Step 3: Define matrix C of size n x n
    v_norm_squared = np.sum(v**2)
    C = np.full((n, n), -v_norm_squared/n)
    np.fill_diagonal(C, v_norm_squared - v_norm_squared/n)
    
    # Step 4: Calculate q vector
    q = f - np.dot(v, e)/n
    
    # Step 5: Solve for lambda
    try:
        lambda_vec = np.zeros(n)
        lambda_vec[:-1] = np.linalg.solve(C[:-1, :-1] + epsilon * np.eye(n-1), q[:-1])
    except np.linalg.LinAlgError as e:
        logger.error(f"Failed to solve linear system in direct projection: {e}")
        raise RuntimeError("Direct projection failed due to singular matrix")
    
    # Step 6: Calculate S
    S = np.sum(lambda_vec)
    
    # Step 7: Calculate eta vector
    eta = (e - S * v)/n
    
    # Step 8: Calculate orthogonal correction
    A_orth = np.outer(eta, np.ones(n)) + np.outer(v, lambda_vec)
    
    # Step 9: Apply correction
    A = A - A_orth
    
    # Step 10: Ensure non-negativity
    A = np.maximum(A, 0)
    
    # Step 11: Normalize rows to ensure partition constraint
    row_sums = np.sum(A, axis=1)
    mask = row_sums > epsilon  # Avoid division by zero
    A[mask] = A[mask] / row_sums[mask, np.newaxis]
    A[~mask] = 1.0/n  # Set uniform distribution for zero rows
    
    # Step 12: Project onto area constraints using iterative refinement
    # This ensures both constraints are satisfied
    for _ in range(5):  # Small number of iterations for refinement
        # Project onto area constraints
        area_sums = v @ A
        scale_factors = d / (area_sums + epsilon)  # Add epsilon to avoid division by zero
        A = A * scale_factors[np.newaxis, :]
        
        # Re-normalize rows to ensure partition constraint
        row_sums = np.sum(A, axis=1)
        mask = row_sums > epsilon  # Avoid division by zero
        A[mask] = A[mask] / row_sums[mask, np.newaxis]
        A[~mask] = 1.0/n  # Set uniform distribution for zero rows
        
        # Check if we're close enough
        area_error = np.max(np.abs(v @ A - d))
        row_error = np.max(np.abs(np.sum(A, axis=1) - 1))
        if area_error < 1e-8 and row_error < 1e-8:
            break
    
    # Validate result
    final_row_error = np.max(np.abs(np.sum(A, axis=1) - 1))
    final_area_error = np.max(np.abs(v @ A - d))

    logger.debug(f"Direct projection completed: row_error={final_row_error:.2e}, area_error={final_area_error:.2e}")

    if _prof is not None:
        _prof.record('projection', time.perf_counter() - t0)

    return A


def _project_simplex_rows(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Euclidean projection of each row of ``Z`` (V x N) onto the
    probability simplex ``{p : sum(p)=1, p>=0}`` (cap-free; the cap at 1 never
    binds when the row sums to 1, see docs/math/08-...).

    Returns ``(U, tau)`` where ``U = maximum(Z - tau[:, None], 0)`` has unit row
    sums and ``tau`` is the per-row threshold, so the row dual is ``alpha = -tau``.
    Sort/threshold form (Held/Duchi/Condat); O(V*N log N), no Python row loop.
    """
    V, N = Z.shape
    Zs = np.sort(Z, axis=1)[:, ::-1]            # descending
    css = np.cumsum(Zs, axis=1) - 1.0
    idx = np.arange(1, N + 1)
    rho = np.count_nonzero(Zs - css / idx > 0, axis=1)   # >= 1 (target sum 1 > 0)
    tau = css[np.arange(V), rho - 1] / rho
    U = np.maximum(Z - tau[:, None], 0.0)
    return U, tau


def orthogonal_projection_newton(A: np.ndarray, c: np.ndarray, d: np.ndarray, v: np.ndarray,
                                 max_iter: int = 500, tol: float = 1e-10,
                                 logger: Optional[logging.Logger] = None,
                                 _prof=None, beta0: Optional[np.ndarray] = None,
                                 return_beta: bool = False,
                                 newton_polish: bool = True) -> np.ndarray:
    """Exact Euclidean projection onto the partition, equal-area, and box
    constraints, computed via the concave dual (docs/math/08-dual-newton-projection).

    Solves, for candidate ``A = Y`` (V x N):
        min_U 1/2||U - Y||^2  s.t.  sum_k U[i,k]=1,  sum_i v_i U[i,k]=d_k,  0<=U<=1.
    The upper box bound is implied by the row constraint with ``U>=0``, so the
    per-vertex inner solve is a probability-simplex projection. The area duals
    ``beta`` are found by maximizing the concave partial dual ``q(beta)`` (gradient
    ``-R``, R the area residual) with L-BFGS-B, optionally polished by a few
    gauge-fixed semismooth-Newton steps with a zero-J-row safeguard.

    Drop-in with :func:`orthogonal_projection_iterative` (same args/return shape).
    ``beta0`` warm-starts the dual (thread it across PGD steps); ``return_beta``
    additionally returns the final ``beta`` for the next warm start.

    Args:
        A: V x N candidate density matrix ``Y``.
        c: length-N row-sum targets (assumed ones, as in the iterative method).
        d: length-N per-cell area targets. Must satisfy ``sum(d) == sum(v)``.
        v: length-V lumped mass (strictly positive).
        max_iter: max L-BFGS-B iterations.
        tol: target max area residual ``|v^T U - d|_inf`` (rows/box are exact).
        beta0: optional warm-start dual (length N); defaults to zeros.
        return_beta: if True, return ``(U, beta)`` instead of ``U``.
        newton_polish: run the Newton polish if L-BFGS leaves area residual > tol.

    Returns:
        The projected V x N matrix ``U`` (or ``(U, beta)`` if ``return_beta``).
    """
    if logger is None:
        logger = get_logger(__name__)
    if _prof is not None:
        t0 = time.perf_counter()
        _prof.add_counter('projection_invocations', 1)

    Y = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    d = np.asarray(d, dtype=float)
    V, N = Y.shape
    if len(c) != N or len(d) != N or len(v) != V:
        raise ValueError(f"Dimension mismatch: A({V}x{N}), c({len(c)}), d({len(d)}), v({len(v)})")

    # Consistency of the equality targets (else the QP is infeasible; math doc Remark 2.2).
    sv = float(v.sum())
    if abs(float(d.sum()) - sv) > 1e-9 * max(sv, 1.0):
        raise ValueError(
            f"Infeasible area targets: sum(d)={d.sum():.6e} != sum(v)={sv:.6e}"
        )

    def _inner(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # z_ik = y_ik + v_i beta_k, then project each row onto the simplex.
        U, tau = _project_simplex_rows(Y + np.outer(v, beta))
        return U, tau

    def _neg_q_and_grad(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        # Minimize -q(beta). At the inner solution rows sum to 1, so
        # q(beta) = 1/2||U-Y||^2 - beta . R, and grad_beta(-q) = R (Danskin).
        U, _ = _inner(beta)
        R = v @ U - d
        neg_q = -0.5 * float(np.sum((U - Y) ** 2)) + float(beta @ R)
        return neg_q, R

    beta = np.zeros(N) if beta0 is None else np.asarray(beta0, dtype=float).copy()

    # L-BFGS only needs to get close enough that the polish's active set is stable;
    # chasing gtol below ~1e-8 just burns iterations (it floors there). The quadratic
    # Newton polish drives the last digits to `tol`.
    lbfgs_gtol = max(tol, 1e-8) if newton_polish else min(tol, 1e-11)
    res = minimize(_neg_q_and_grad, beta, jac=True, method='L-BFGS-B',
                   options={'maxiter': int(max_iter), 'gtol': lbfgs_gtol,
                            'ftol': 1e-16, 'maxcor': 20})
    beta = res.x
    n_inner = int(res.nit)

    U = _inner(beta)[0]
    R = v @ U - d
    area_err = float(np.max(np.abs(R))) if R.size else 0.0

    # Optional semismooth-Newton polish on the concave dual (docs/math/08 sec 8):
    # gauge-fixed via the min-norm lstsq (kernel span{1}), q-ascent line search,
    # zero-J-row safeguard (a zero row leaves that coordinate untouched -- lstsq
    # never explodes, unlike an eps-ridge). Only the last digits of feasibility.
    if newton_polish and area_err > tol:
        v2 = v ** 2
        for _ in range(30):
            if area_err <= tol:
                break
            mask = (U > 1e-12).astype(float)                 # active-set indicator a_i
            m = mask.sum(axis=1)                             # |A_i| >= 1
            J = np.diag(v2 @ mask) - mask.T @ ((v2 / m)[:, None] * mask)   # eq:jacobian
            # min-norm lstsq gauge-fixes (kernel span{1}); on a zero J-row it leaves
            # that coordinate at 0 -- no eps-ridge explosion (docs/math/08 Remark 8.1).
            delta, *_ = np.linalg.lstsq(J, -R, rcond=None)
            # Newton-for-roots on R(beta)=0: backtrack on the area residual itself
            # (near the flat concave max a q-merit cannot resolve the last digits).
            step, accepted = 1.0, False
            for _ in range(40):
                U_try, _ = _inner(beta + step * delta)
                R_try = v @ U_try - d
                if np.max(np.abs(R_try)) < area_err:
                    beta, U, R = beta + step * delta, U_try, R_try
                    area_err = float(np.max(np.abs(R)))
                    n_inner += 1
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break

    row_err = float(np.max(np.abs(U.sum(axis=1) - 1.0)))
    nonneg_err = max(0.0, -float(U.min()))
    if area_err > 10 * tol or row_err > 1e-9 or nonneg_err > 1e-12:
        logger.warning(
            f"Newton projection residuals above target: row={row_err:.2e}, "
            f"area={area_err:.2e}, nonneg={nonneg_err:.2e} (tol={tol:.1e})"
        )

    if _prof is not None:
        _prof.add_counter('projection_inner_iters_total', n_inner)
        _prof.record('projection', time.perf_counter() - t0)

    return (U, beta) if return_beta else U


def validate_projection_result(A: np.ndarray, v: np.ndarray, d: np.ndarray,
                             tol: float = 1e-8, logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a projected matrix satisfies the constraints.
    
    Args:
        A: Projected matrix
        v: Mass matrix column sums
        d: Target area constraints
        tol: Tolerance for validation
        logger: Logger instance
        
    Returns:
        True if constraints are satisfied within tolerance
    """
    if logger is None:
        logger = get_logger(__name__)
    
    # Check partition constraint: each row should sum to 1
    row_sums = np.sum(A, axis=1)
    row_error = np.max(np.abs(row_sums - 1))
    
    # Check area constraint: v^T A should equal d
    area_sums = v @ A
    area_error = np.max(np.abs(area_sums - d))
    
    # Check non-negativity
    min_val = np.min(A)
    non_neg_error = max(0, -min_val)
    
    logger.debug(f"Projection validation: row_error={row_error:.2e}, "
                f"area_error={area_error:.2e}, non_neg_error={non_neg_error:.2e}")
    
    return row_error < tol and area_error < tol and non_neg_error < tol

def create_initial_condition_with_projection(N: int, n_partitions: int, v: np.ndarray,
                                           seed: int = None, method: str = "iterative",
                                           max_iter: int = 100, tol: float = 1e-8,
                                           logger: Optional[logging.Logger] = None,
                                           _prof=None) -> np.ndarray:
    """
    Create a valid initial condition using orthogonal projection.
    
    Args:
        N: Number of vertices
        n_partitions: Number of partitions
        v: Mass matrix column sums
        seed: Random seed for reproducibility
        method: "iterative" or "direct"
        max_iter: Maximum iterations for iterative method
        tol: Tolerance for convergence
        logger: Logger instance
        
    Returns:
        Valid initial condition vector
    """
    if logger is None:
        logger = get_logger(__name__)
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random initial condition
    x0 = np.random.rand(N * n_partitions)
    A = x0.reshape(N, n_partitions)
    
    # Define constraints
    c = np.ones(n_partitions)  # Row sums should be 1
    d = np.sum(v) / n_partitions * np.ones(n_partitions)  # Equal areas
    
    logger.info(f"Creating initial condition: {N}x{n_partitions}, method={method}")
    
    # Apply projection
    if method == "iterative":
        A_projected = orthogonal_projection_iterative(A, c, d, v, max_iter, tol, logger, _prof=_prof)
    elif method == "direct":
        A_projected = orthogonal_projection_direct(A, c, d, v, logger, _prof=_prof)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    # Validate result
    if not validate_projection_result(A_projected, v, d, tol, logger):
        logger.warning("Projection validation failed, but returning result anyway")
    
    return A_projected.flatten() 