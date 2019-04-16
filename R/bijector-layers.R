#' A Masked Autoencoder for Distribution Estimation
#'
#'  A `AutoregressiveLayer` takes as input a Tensor of shape `[..., event_size]`
#' and returns a Tensor of shape `[..., event_size, params]`.
#' The output satisfies the autoregressive property.  That is, the layer is
#' configured with some permutation `ord` of `{0, ..., event_size-1}` (i.e., an
#' ordering of the input dimensions), and the output `output[batch_idx, i, ...]`
#' for input dimension `i` depends only on inputs `x[batch_idx, j]` where
#' `ord(j) < ord(i)`.  The autoregressive property allows us to use
#' `output[batch_idx, i]` to parameterize conditional distributions:
#'   `p(x[batch_idx, i] | x[batch_idx, ] for ord(j) < ord(i))`
#' which give us a tractable distribution over input `x[batch_idx]`:
#'   `p(x[batch_idx]) = prod_i p(x[batch_idx, ord(i)] | x[batch_idx, ord(0:i)])`
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_distribution_lambda
#'
#' @param params Integer specifying the number of parameters to output per input.
#' @param event_shape List of positive integers (or a single int),
#'  specifying the shape of the input to this layer, which is also the
#'  event_shape of the distribution parameterized by this layer.  Currently
#'  only rank-1 shapes are supported.  That is, event_shape must be a single
#'  integer.  If not specified, the event shape is inferred when this layer
#'  is first called or built.
#' @param hidden_units List of non-negative integers, specifying the number of
#'  units in each hidden layer.
#' @param input_order Order of degrees to the input units: 'random',
#'  left-to-right', 'right-to-left', or an array of an explicit order. For
#'  example, 'left-to-right' builds an autoregressive model:
#'    `p(x) = p(x1) p(x2 | x1) ... p(xD | x<D)`.  Default: 'left-to-right'.
#' @param hidden_degrees Method for assigning degrees to the hidden units:
#'  equal', 'random'.  If 'equal', hidden units in each layer are allocated
#'  equally (up to a remainder term) to each degree.  Default: 'equal'.
#' @param validate_args  Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#'
#' @export
layer_autoregressive <- function(object,
                                 params,
                                 event_shape = NULL,
                                 hidden_units = NULL,
                                 input_order = c("left-to-right", "right-to-left", "random"),
                                 hidden_degrees = c("equal", "random"),
                                 activation = NULL,
                                 use_bias = TRUE,
                                 kernel_initializer = "glorot_uniform",
                                 validate_args = FALSE,
                                 batch_input_shape = NULL,
                                 input_shape = NULL,
                                 batch_size = NULL,
                                 dtype = NULL,
                                 name = NULL,
                                 trainable = NULL,
                                 weights = NULL) {
  args <- list(
    params = params,
    event_shape = normalize_shape(event_shape),
    hidden_units = hidden_units,
    input_order = match.arg(input_order),
    hidden_degrees = match.arg(hidden_degrees),
    activation = activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    validate_args = validate_args,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )

  create_layer(
    tfp$bijectors$masked_autoregressive$AutoregressiveLayer,
    object,
    args
  )
}
