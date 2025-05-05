---
title: "assignment-data-generator"
author: "Venkata Sai Pallavi Pallapolu"
date: "2025-05-05"
output: html_document
---
```{r}
install.packages("xgboost")
install.packages("caret")
install.packages("dplyr")
install.packages("purrr")
install.packages("mlbench")
```

```{r}
library(microbenchmark)
library(xgboost)
library(caret)
library(dplyr)
library(mlbench)
library(purrr)
```

```{r}
data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)
```

```{r}
cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames
```

```{r}
sz <- 10000000 ## to be used in sampling
sample(ds$pregnant, size = sz, replace = T)
```

```{r}
dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe
```

```{r}
names(dfdata) <- prednames
dfdata
```

```{r}
class(cfs[2:length(cfs)])
```

```{r}
length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$", prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept
```

```{r}
## exponentiate the logit to obtain probability values of thee outcome variable
dfdata['outcome'] <- ifelse(1/(1 + exp(-(pvec))) > 0.5, 1, 0)
```

```{r}
write.csv(dfdata, "dfdata_10M.csv", row.names = FALSE)
```
