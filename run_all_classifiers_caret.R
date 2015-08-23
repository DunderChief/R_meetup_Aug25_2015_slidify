options(stringsAsFactors = FALSE)
library(caret)
library(stringr)
library(foreach)
library(doMC)
registerDoMC(7)
models <- read.csv('caret_models.csv')
table(models$Type)

# Number of packages
getReqPackages <- function(package_list) {
  packs <- paste0(models$Packages, collapse=', ')
  packs <- str_split(packs, ', ')[[1]]
  packs <- packs[packs!='']
  packs <- unique(packs)
  return(packs)
}

all_packs <- getReqPackages(models$Packages)
length(unique(all_packs))

class_models <- subset(models, Type %in% c('Classification', 'Dual Use'))
class_packs <- getReqPackages(class_models$Packages)
length(unique(class_packs))

do.call(c, 
lapply(class_packs, function(this.package){
  if(!require(MASS)){
    return(this.package)
  } else {return(1)}
})
)

Sys.time()
myFits <- foreach(this.model = class_models$method.Argument, .errorhandling='pass') %do% {
  print(this.model)
  timing <- system.time(
    fit <- train(Species ~ ., 
                 data=iris,
                 method=this.model,
                 preProcess='pca',
                 trControl=trainControl(method='repeatedcv', number=7, repeats=1),
                 tuneLength=5)
  )
  return(list(fit, timing))
}
Sys.time()
names(myFits) <- class_models$method.Argument
confMats <- lapply(myFits, function(xx) confusionMatrix(xx[[1]]))
