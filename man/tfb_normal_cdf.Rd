% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bijectors.R
\name{tfb_normal_cdf}
\alias{tfb_normal_cdf}
\title{Compute \code{Y = g(X) = NormalCDF(x)}.}
\usage{
tfb_normal_cdf(validate_args = FALSE, name = "normal")
}
\arguments{
\item{validate_args}{Logical, default FALSE. Whether to validate input with asserts. If validate_args is
FALSE, and the inputs are invalid, correct behavior is not guaranteed.}

\item{name}{name prefixed to Ops created by this class.}
}
\description{
This bijector maps inputs from \code{[-inf, inf]} to \code{[0, 1]}. The inverse of the
bijector applied to a uniform random variable \code{X ~ U(0, 1)} gives back a
random variable with the \href{https://en.wikipedia.org/wiki/Normal_distribution}{Normal distribution}:
}
\details{
\code{Y ~ Normal(0, 1)}
\code{pdf(y; 0., 1.) = 1 / sqrt(2 * pi) * exp(-y ** 2 / 2)}
}
\seealso{
Other bijectors: \code{\link{masked_autoregressive_default_template}},
  \code{\link{masked_dense}},
  \code{\link{real_nvp_default_template}},
  \code{\link{tfb_absolute_value}},
  \code{\link{tfb_affine_linear_operator}},
  \code{\link{tfb_affine_scalar}},
  \code{\link{tfb_affine}},
  \code{\link{tfb_batch_normalization}},
  \code{\link{tfb_blockwise}}, \code{\link{tfb_chain}},
  \code{\link{tfb_cholesky_outer_product}},
  \code{\link{tfb_cholesky_to_inv_cholesky}},
  \code{\link{tfb_discrete_cosine_transform}},
  \code{\link{tfb_expm1}}, \code{\link{tfb_exp}},
  \code{\link{tfb_fill_triangular}},
  \code{\link{tfb_gumbel}}, \code{\link{tfb_identity}},
  \code{\link{tfb_inline}}, \code{\link{tfb_invert}},
  \code{\link{tfb_kumaraswamy}},
  \code{\link{tfb_masked_autoregressive_flow}},
  \code{\link{tfb_matrix_inverse_tri_l}},
  \code{\link{tfb_matvec_lu}}, \code{\link{tfb_ordered}},
  \code{\link{tfb_permute}},
  \code{\link{tfb_power_transform}},
  \code{\link{tfb_real_nvp}}, \code{\link{tfb_reciprocal}},
  \code{\link{tfb_reshape}}, \code{\link{tfb_scale_tri_l}},
  \code{\link{tfb_sigmoid}},
  \code{\link{tfb_sinh_arcsinh}},
  \code{\link{tfb_softmax_centered}},
  \code{\link{tfb_softplus}}, \code{\link{tfb_softsign}},
  \code{\link{tfb_square}}, \code{\link{tfb_tanh}},
  \code{\link{tfb_transform_diagonal}},
  \code{\link{tfb_transpose}}, \code{\link{tfb_weibull}}
}
\concept{bijectors}
