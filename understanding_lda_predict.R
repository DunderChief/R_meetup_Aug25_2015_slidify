

# function (object, newdata, prior = object$prior, dimen, method = c("plug-in", 
#      "predictive", "debiased"), ...) 
library(MASS)
fit <- lda(Species ~ Petal.Length , data=iris[-55, ], prior=rep(1/3, 3))
predict(fit, newdata=iris[55, ])

object <- fit
prior <- fit$prior
newdata <- iris[55, ]


Terms <- object$terms
Terms <- delete.response(Terms)

newdata <- model.frame(Terms, newdata, na.action = na.pass, 
                       xlev = object$xlevels)

cl <- attr(Terms, "dataClasses") 
.checkMFClasses(cl, newdata)

x <- model.matrix(Terms, newdata, contrasts = object$contrasts)
xint <- match("(Intercept)", colnames(x), nomatch = 0L)

x <- x[, -xint, drop = FALSE]

means <- colSums(prior * object$means)
scaling <- object$scaling
x <- scale(x, center = means, scale = FALSE) %*% scaling
dm <- scale(object$means, center = means, scale = FALSE) %*% scaling
method <- 'plug-in'
dimen <-  length(object$svd)
N <- object$N

ng <- length(object$prior)

dm <- dm[, 1L:dimen, drop = FALSE]
dist <- matrix(0.5 * rowSums(dm^2) - log(prior), nrow(x), 
               length(prior), byrow = TRUE) - x[, 1L:dimen, drop = FALSE] %*% t(dm)
dist <- exp(-(dist - apply(dist, 1L, min, na.rm = TRUE)))

posterior <- dist/drop(dist %*% rep(1, ng))
