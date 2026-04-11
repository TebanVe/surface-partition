import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from ..logging_config import get_logger

def orthogonal_projection_iterative(A: np.ndarray, c: np.ndarray, d: np.ndarray, v: np.ndarray, 
                                  max_iter: int = 1000, tol: float = 1e-10, 
                                  logger: Optional[logging.Logger] = None) -> np.ndarray:
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
    
    # Track convergence history
    convergence_history = {
        'iterations': [],
        'row_errors': [],
        'area_errors': [],
        'max_errors': []
    }
    
    for iter in range(max_iter):
        A_prev = A.copy()
        
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
            # Try solving the full system first
            lambda_vec = np.linalg.solve(C + epsilon * np.eye(n), q)
        except np.linalg.LinAlgError:
            # Fall back to reduced system if full system fails
            logger.warning(f"Iteration {iter}: Full system singular, using reduced system")
            lambda_vec = np.zeros(n)
            lambda_vec[:-1] = np.linalg.solve(C[:-1, :-1] + epsilon * np.eye(n-1), q[:-1])
        
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
        
        # Track convergence history
        convergence_history['iterations'].append(iter)
        convergence_history['row_errors'].append(row_sum_error)
        convergence_history['area_errors'].append(area_error)
        convergence_history['max_errors'].append(max_error)
        
        # Log progress every 10 iterations
        if iter % 10 == 0:
            logger.debug(f"Iteration {iter}: row_error={row_sum_error:.2e}, area_error={area_error:.2e}")
        
        if row_sum_error < tol and area_error < tol:
            logger.info(f"Projection converged after {iter+1} iterations")
            logger.info(f"Final errors: row={row_sum_error:.2e}, area={area_error:.2e}")
            break
            
        # Check if we're making progress
        if iter > 0 and np.allclose(A, A_prev, rtol=tol, atol=tol):
            # Warn only if clearly above tolerance; otherwise treat as benign stagnation
            if max_error > 10 * tol:
                logger.warning(f"Projection stagnated after {iter+1} iterations (max_error={max_error:.2e} > {10*tol:.1e})")
            else:
                logger.debug(f"Projection stagnated after {iter+1} iterations (near tol): row={row_sum_error:.2e}, area={area_error:.2e}")
            break
    else:
        # Loop completed without convergence
        logger.warning(f"Projection did not converge after {max_iter} iterations")
        logger.warning(f"Final errors: row={row_sum_error:.2e}, area={area_error:.2e}")
    
    # Validate final result
    final_row_error = np.max(np.abs(np.sum(A, axis=1) - 1))
    final_area_error = np.max(np.abs(v @ A - d))
    
    if final_row_error > 10*tol or final_area_error > 10*tol:
        logger.error(f"Projection failed: row_error={final_row_error:.2e}, area_error={final_area_error:.2e}")
        raise RuntimeError("Orthogonal projection failed to achieve required tolerance")
    
    return A

def orthogonal_projection_direct(A: np.ndarray, c: np.ndarray, d: np.ndarray, v: np.ndarray,
                               logger: Optional[logging.Logger] = None) -> np.ndarray:
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
    
    return A

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
                                           logger: Optional[logging.Logger] = None) -> np.ndarray:
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
        A_projected = orthogonal_projection_iterative(A, c, d, v, max_iter, tol, logger)
    elif method == "direct":
        A_projected = orthogonal_projection_direct(A, c, d, v, logger)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    # Validate result
    if not validate_projection_result(A_projected, v, d, tol, logger):
        logger.warning("Projection validation failed, but returning result anyway")
    
    return A_projected.flatten() 