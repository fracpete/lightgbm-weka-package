/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * LightGBMUtils.java
 * Copyright (C) 2023 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions;

import io.github.metarank.lightgbm4j.LGBMDataset;
import io.github.metarank.lightgbm4j.LGBMException;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Utility functions for LightGBM.
 *
 * @author fracpete (fracpete at waikato dot ac dot nz)
 */
public class LightGBMUtils {

  /**
   * Converts the Weka Instances into a LightGBM dataset.
   *
   * @param data	the data to convert
   * @return		the generated dataset
   * @throws LGBMException	if conversion fails
   */
  public static LGBMDataset fromInstances(Instances data) throws LGBMException {
    LGBMDataset	result;
    int		clsIndex;
    String[]	columns;
    double[] 	attValues;
    float[] 	clsValues;
    int		i;
    int		n;
    int		offset;
    double[]	values;

    clsIndex = data.classIndex();

    // attribute names
    columns = new String[data.numAttributes() - (clsIndex == -1 ? 0 : 1)];
    n = 0;
    for (i = 0; i < data.numAttributes(); i++) {
      if (i == clsIndex)
        continue;
      columns[n] = data.attribute(i).name();
      n++;
    }

    // class values
    clsValues = null;
    if (clsIndex > -1) {
      values    = data.attributeToDoubleArray(clsIndex);
      clsValues = new float[values.length];
      for (i = 0; i < values.length; i++)
        clsValues[i] = (float) values[i];
    }

    // attribute values
    attValues = new double[data.numInstances() * columns.length];
    offset    = 0;
    for (Instance inst: data) {
      n = 0;
      values = inst.toDoubleArray();
      for (i = 0; i < values.length; i++) {
	if (i == clsIndex)
	  continue;
	attValues[offset + n] = values[i];
	n++;
      }
      offset += columns.length;
    }

    // create dataset
    result = LGBMDataset.createFromMat(attValues, data.numInstances(), columns.length, true, "", null);
    result.setFeatureNames(columns);
    if (clsValues != null)
      result.setField("label", clsValues);

    return result;
  }

  /**
   * Converts the Weka Instance into a double array for LightGBM (excluding class value).
   *
   * @param data	the data to convert
   * @return		the generated dataset
   */
  public static double[] fromInstance(Instance data) {
    double[]	result;
    int		clsIndex;
    int		n;
    int		i;

    clsIndex = data.classIndex();

    result = new double[data.numAttributes() - (clsIndex == -1 ? 0 : 1)];
    n = 0;
    for (i = 0; i < data.numAttributes(); i++) {
      if (i == clsIndex)
	continue;
      result[n] = data.value(i);
      n++;
    }

    return result;
  }
}
