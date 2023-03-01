# lightgbm-weka-package

Weka package for classifier (regression/classification) using the
[LightGBM gradient boosting framework](https://github.com/microsoft/LightGBM).


## Parameters

```
weka.classifiers.functions.LightGBM

-O <REGRESSION|REGRESSION_L1|HUBER|FAIR|POISSON|QUANTILE|MAPE|GAMMA|TWEEDIE|BINARY|MULTICLASS|MULTICLASSOVA|CROSSENTROPY|CROSSENTROPY_LAMBDA|LAMBDA_RANK|RANK_XENDCG>
	The type of booster to use:
	REGRESSION = Regression
	REGRESSION_L1 = Regression L1
	HUBER = Huber loss
	FAIR = Fair loss
	POISSON = Poisson regression
	QUANTILE = Quantile regression
	MAPE = MAPE loss
	GAMMA = Gamma regression with log-link
	TWEEDIE = Tweedie regression with log-link
	BINARY = Binary log loss classification
	MULTICLASS = Multi-class (softmax)
	MULTICLASSOVA = Multi-class (one-vs-all)
	CROSSENTROPY = Cross-entropy
	CROSSENTROPY_LAMBDA = Cross-entropy Lambda
	LAMBDA_RANK = Lambda rank
	RANK_XENDCG = Rank Xendcg
	(default: REGRESSION)
-P <parameters>
	The parameters for the booster (blank-separated key=value pairs).
	See: https://lightgbm.readthedocs.io/en/v3.3.2/Parameters.html
	(default: none)

-I <iterations>
	The number of iterations to train for.
	(default: 1000)

-V <0-100>
	The size of the validation set to split off from the training set.
	(default: 0.0)

-R
	Turns on randomization before splitting off the validation set.
	(default: off)
```


## Releases

* [2023.3.1](https://github.com/fracpete/lightgbm-weka-package/releases/download/v2023.3.1/lightgbm-2023.3.1.zip)
* [2023.2.10](https://github.com/fracpete/lightgbm-weka-package/releases/download/v2023.2.10/lightgbm-2023.2.10.zip)


## Maven

Use the following dependency in your `pom.xml`:

```xml
    <dependency>
      <groupId>com.github.fracpete</groupId>
      <artifactId>lightgbm-weka-package</artifactId>
      <version>2023.3.1</version>
      <type>jar</type>
      <exclusions>
        <exclusion>
          <groupId>nz.ac.waikato.cms.weka</groupId>
          <artifactId>weka-dev</artifactId>
        </exclusion>
      </exclusions>
    </dependency>
```


## How to use packages

For more information on how to install the package, see:

https://waikato.github.io/weka-wiki/packages/manager/


