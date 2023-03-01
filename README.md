# lightgbm-weka-package

Weka package for classifier (regression/classification) using the
[LightGBM gradient boosting framework](https://github.com/microsoft/LightGBM).


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


