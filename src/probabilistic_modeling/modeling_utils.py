import torch


def covariance_output_to_cholesky(pred_bbox_cov):
    """
    Transforms output to covariance cholesky decomposition.
    Args:
        pred_bbox_cov (kx4 or kx10): Output covariance matrix elements.

    Returns:
        predicted_cov_cholesky (kx4x4): cholesky factor matrix
    """
    # Embed diagonal variance
    diag_vars = torch.sqrt(torch.exp(pred_bbox_cov[:, 0:4]))
    predicted_cov_cholesky = torch.diag_embed(diag_vars)

    if pred_bbox_cov.shape[1] > 4:
        tril_indices = torch.tril_indices(row=4, col=4, offset=-1)
        predicted_cov_cholesky[:, tril_indices[0],
                               tril_indices[1]] = pred_bbox_cov[:, 4:]

    return predicted_cov_cholesky
