---
title       : Predictive Analytics in R
subtitle    : 
author      : David O'Brien <dunder.chief@gmail.com>
job         : 
framework   : revealjs        # {io2012, html5slides, shower, dzslides, ...}
revealjs    : {theme:      default, 
               transition: concave}
highlighter : prettify  # {highlight.js, prettify, highlight}
hitheme     : desert
widgets     : []            # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}
knit        : slidify::knit2slides
---

# Predictive Analytics in R
### August 25, 2015


--- &vertical

## Read-And-Delete

1. Edit YAML front matter
2. Write using R Markdown
3. Use an empty line followed by three dashes to separate slides!

--- &vertical

## Slide 2


```r
plot(1:10, rnorm(10, 10))
```

![plot of chunk unnamed-chunk-1](assets/fig/unnamed-chunk-1-1.png) 


---

<!-----ML Overview---------------------------------------------->
<!---1--->
What is Predictive Modeling?
------------------------------------------------------------

1. Given a set of **predictor variables (X)**, 

2. Predict an **outcome (Y)**


<table>
 <thead>
  <tr>
   <th style="text-align:center;"> Sepal Length
[X1] </th>
   <th style="text-align:center;"> Sepal Width
[X2] </th>
   <th style="text-align:center;"> Petal Length
[X3] </th>
   <th style="text-align:center;"> Petal Width
[X4] </th>
   <th style="text-align:center;"> Species
[Y] </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:center;"> 6.5 </td>
   <td style="text-align:center;"> 2.8 </td>
   <td style="text-align:center;"> 4.6 </td>
   <td style="text-align:center;"> 1.5 </td>
   <td style="text-align:center;"> ??? </td>
  </tr>
</tbody>
</table>

<br>

<div class="centered">
<img src='img/iris.png' width="700">
</div>

<div class="notes">
Machine learning / Predictive analytics / predictive modeling / Statistical learning / etc.

variable / feature / covariate / predictors

</div>

<div class='notes'>

these are notes
</div>

--- &vertical

<!---2--->
Our guess: 
---------------------------------------------------------------------

<table>
 <thead>
  <tr>
   <th style="text-align:center;"> Sepal Length
[X1] </th>
   <th style="text-align:center;"> Sepal Width
[X2] </th>
   <th style="text-align:center;"> Petal Length
[X3] </th>
   <th style="text-align:center;"> Petal Width
[X4] </th>
   <th style="text-align:center;"> Species
[Y] </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:center;"> 6.5 </td>
   <td style="text-align:center;"> 2.8 </td>
   <td style="text-align:center;"> 4.6 </td>
   <td style="text-align:center;"> 1.5 </td>
   <td style="text-align:center;"> ??? </td>
  </tr>
</tbody>
</table>
<br> <!---- Arrow Image --->
<div class="centered"> <img src='img/down.png' width="50"> </div>
<br>

_Equation_

_Equation filled in_

--- &vertical

<!---3--->
How do we estimate these parameters: 
-----------------------------

<table>
 <thead>
  <tr>
   <th style="text-align:center;"> Sepal Length
[X1] </th>
   <th style="text-align:center;"> Sepal Width
[X2] </th>
   <th style="text-align:center;"> Petal Length
[X3] </th>
   <th style="text-align:center;"> Petal Width
[X4] </th>
   <th style="text-align:center;"> Species
[Y] </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:center;"> 5.1 </td>
   <td style="text-align:center;"> 3.5 </td>
   <td style="text-align:center;"> 1.4 </td>
   <td style="text-align:center;"> 0.2 </td>
   <td style="text-align:center;"> setosa </td>
  </tr>
  <tr>
   <td style="text-align:center;"> 4.9 </td>
   <td style="text-align:center;"> 3.0 </td>
   <td style="text-align:center;"> 1.4 </td>
   <td style="text-align:center;"> 0.2 </td>
   <td style="text-align:center;"> setosa </td>
  </tr>
  <tr>
   <td style="text-align:center;"> 7.0 </td>
   <td style="text-align:center;"> 3.2 </td>
   <td style="text-align:center;"> 4.7 </td>
   <td style="text-align:center;"> 1.4 </td>
   <td style="text-align:center;"> versicolor </td>
  </tr>
  <tr>
   <td style="text-align:center;"> 6.4 </td>
   <td style="text-align:center;"> 3.2 </td>
   <td style="text-align:center;"> 4.5 </td>
   <td style="text-align:center;"> 1.5 </td>
   <td style="text-align:center;"> versicolor </td>
  </tr>
  <tr>
   <td style="text-align:center;"> 6.3 </td>
   <td style="text-align:center;"> 3.3 </td>
   <td style="text-align:center;"> 6.0 </td>
   <td style="text-align:center;"> 2.5 </td>
   <td style="text-align:center;"> virginica </td>
  </tr>
  <tr>
   <td style="text-align:center;"> 5.8 </td>
   <td style="text-align:center;"> 2.7 </td>
   <td style="text-align:center;"> 5.1 </td>
   <td style="text-align:center;"> 1.9 </td>
   <td style="text-align:center;"> virginica </td>
  </tr>
</tbody>
</table>

Use this _historical_ data to optimize the best fit for our future models:

We use this dataset to find:

1. Mean for each measurements for each class
2. 
3. 

--- 

<!---4--->
Implementation in R: 
------------------------------------------


```r
library(MASS)
trainset <- iris[-example_row, ] 
fit.lda <- lda(Species ~ ., data=trainset, prior=c(1/3, 1/3, 1/3)) 
pred <- predict(fit.lda, iris[example_row, ])
round(pred$posterior, 3)
```

```
##    setosa versicolor virginica
## 55      0      0.995     0.005
```


```r
kable(head(iris))
```



| Sepal.Length| Sepal.Width| Petal.Length| Petal.Width|Species |
|------------:|-----------:|------------:|-----------:|:-------|
|          5.1|         3.5|          1.4|         0.2|setosa  |
|          4.9|         3.0|          1.4|         0.2|setosa  |
|          4.7|         3.2|          1.3|         0.2|setosa  |
|          4.6|         3.1|          1.5|         0.2|setosa  |
|          5.0|         3.6|          1.4|         0.2|setosa  |
|          5.4|         3.9|          1.7|         0.4|setosa  |

