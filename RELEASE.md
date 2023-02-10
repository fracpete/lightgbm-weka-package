How to make a release
=====================

Switched to Java 8.

Preparation
-----------

* Change the artifact ID in `pom.xml` to today's date, e.g.:

  ```
  2023.2.10-SNAPSHOT
  ```

* Update the version, date and URL in `Description.props` to reflect new
  version, e.g.:

  ```
  Version=2023.2.10
  Date=2023-02-10
  PackageURL=https://github.com/fracpete/lightgbm-weka-package/releases/download/v2023.2.10/lightgbm-2023.2.10.zip
  ```

* Commit/push all changes


Weka package
------------

* Run the following command to generate the package archive for version `2023.2.10`:

  ```
  ant -f build_package.xml -Dpackage=lightgbm-2023.2.10 clean make_package
  ```

* Create a release tag on github (v2023.2.10)
* add release notes
* upload package archive from `dist`


Maven
-----

* Run the following command to deploy the artifact:

  ```
  mvn release:clean release:prepare release:perform
  ```

* log into https://oss.sonatype.org and close/release artifacts

* After successful deployment, push the changes out:

  ```
  git push
  ````

