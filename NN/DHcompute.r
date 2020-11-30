#' @export
compute <-
function (x, covariate, rep = 1, response=NULL,exposure=NULL) 
{
    nn  <- x
    linear.output <- nn$linear.output
    nrow.weights  <- sapply(nn$weights[[1]], nrow)
    ncol.weights  <- sapply(nn$weights[[1]], ncol)
    weights       <- unlist(nn$weights)
    if (any(is.na(weights))) 
    weights[is.na(weights)] <- 0
    weights        <- relist(weights, nrow.weights, ncol.weights)
    length.weights <- length(weights)
    covariate      <- as.matrix(cbind(1, covariate))
    act.fct        <- nn$act.fct
    neurons        <- list(covariate)
    
    if (length.weights > 1) 
        for (i in 1:(length.weights - 1)) {
            temp <- neurons[[i]] %*% weights[[i]]
            act.temp <- act.fct(temp)
            neurons[[i + 1]] <- cbind(1, act.temp)
        }
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    if (linear.output) 
        net.result  <- temp
    else net.result <- act.fct(temp)
    
    #------------------------Poisson Deviance-------------------------------
    type.err<-type(nn$err.fct)
    if (type.err=="poisson")
    {
       net.result = exp(net.result)
       if (!is.null(response) && !is.null(exposure))
       {
          y       = response
          e       = exposure
          lny     = log(y) 
          lny[is.infinite(lny)]=0
          yhat    = net.result
          lnyhat  = log(yhat) 
          devstat = 2*e*(y*lny-y*lnyhat-y+yhat)
          mse     = e*(y-yhat)^2
       } 
       else 
       { 
          devstat=0
          mse    =0
       }
      
    } #endif poisson
    #------------------------Gaussian Deviance-------------------------------
    if (type.err=="gaussian")
    {
      if (!is.null(response) && !is.null(exposure))
      {
        y       = response
        e       = exposure
        yhat    = net.result
        devstat = e*(y-yhat)^2
        mse     = e*(y-yhat)^2
      } 
      else 
      { 
        devstat=0
        mse    =0
      }
      
    } #endif gaussian
    #------------------------Gamma Deviance-------------------------------
    if (type.err=="gamma")
    {
      net.result = exp(net.result)
      if (!is.null(response) && !is.null(exposure))
      {
        y       = response
        e       = exposure
        lny     = log(y) 
        yhat    = net.result
        lny[is.infinite(lny)]=0
        lnyhat  = log(yhat) 
        devstat = 2*e*(y/yhat-1-log(y/yhat))   #gamma deviance
        mse     = e*(y-yhat)^2
      } 
      else 
      { 
        devstat=0
        mse    =0
      }
    } #endif gamma
    #------------------------Binomial Deviance-------------------------------
    if (type.err=="binomial")
    {
       net.result = 1/(1+exp(net.result)) 
       if (!is.null(response) && !is.null(exposure))
       {
         y       = response
         e       = exposure
         yhat    = net.result
         lny     = log(y) 
         ln1my   = log(1-y)
         lny[is.infinite(lny)]     = 0
         ln1my[is.infinite(ln1my)] = 0
         devstat = 2*e*(y*lny-y*log(yhat)+(1-y)*ln1my-(1-y)*log(1-yhat)) #binomial deviance  
         mse     = e*(y-yhat)^2
       } 
       else 
       { 
         devstat=0
         mse    =0
       }
    }
    #---------------------------------------------------------------  
    list(neurons = neurons, net.result = net.result,devstat = sum(devstat),mse=mean(mse))
}
