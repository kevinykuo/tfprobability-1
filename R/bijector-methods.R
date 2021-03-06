#' Returns the forward Bijector evaluation, i.e., `X = g(Y)`.
#'
#' @param bijector The bijector to apply
#' @param x Tensor. The input to the "forward" evaluation.
#' @family bijector_methods
#' @export
tfb_forward <-
  function(bijector, x, name ="forward") {
    bijector$forward(as_tf_float(x), name)
  }

#' Returns the inverse Bijector evaluation, i.e., `X = g^{-1}(Y)`.
#'
#' @param bijector  The bijector to apply
#' @param y Tensor. The input to the "inverse" evaluation.
#' @family bijector_methods
#' @export
tfb_inverse <-
  function(bijector, y, name="inverse") {
    bijector$inverse(as_tf_float(y), name)
  }

#' Returns both the forward_log_det_jacobian.
#'
#' @param bijector The bijector to apply
#' @param x Tensor. The input to the "forward" Jacobian determinant evaluation.
#' @param event_ndims Number of dimensions in the probabilistic events being transformed.
#'  Must be greater than or equal to bijector$forward_min_event_ndims. The result is summed over the final
#'  dimensions to produce a scalar Jacobian determinant for each event, i.e. it has shape
#'   x$shape$ndims - event_ndims dimensions.
#' @family bijector_methods
#' @export
tfb_forward_log_det_jacobian <-
  function(bijector, x, event_ndims, name="forward_log_det_jacobian") {
    bijector$forward_log_det_jacobian(as_tf_float(x), as.integer(event_ndims), name)
  }

#' Returns the `(log o det o Jacobian o inverse)(y)`.
#'
#' @param bijector The bijector to apply
#' @param y Tensor. The input to the "inverse" Jacobian determinant evaluation.
#' @param event_ndims Number of dimensions in the probabilistic events being transformed.
#'  Must be greater than or equal to bijector$inverse_min_event_ndims. The result is summed over the final
#'  dimensions to produce a scalar Jacobian determinant for each event, i.e. it has shape
#'   x$shape$ndims - event_ndims dimensions.
#' @family bijector_methods
#' @export
tfb_inverse_log_det_jacobian <-
  function(bijector, y, event_ndims, name="inverse_log_det_jacobian") {
    bijector$inverse_log_det_jacobian(as_tf_float(y), as.integer(event_ndims), name)
  }

