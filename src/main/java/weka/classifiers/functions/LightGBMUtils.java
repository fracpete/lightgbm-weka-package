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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

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
    return fromInstances(data, null);
  }

  /**
   * Converts the Weka Instances into a LightGBM dataset.
   *
   * @param data	the data to convert
   * @param reference   the reference dataset to use
   * @return		the generated dataset, can be null
   * @throws LGBMException	if conversion fails
   */
  public static LGBMDataset fromInstances(Instances data, LGBMDataset reference) throws LGBMException {
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
    result = LGBMDataset.createFromMat(attValues, data.numInstances(), columns.length, true, "", reference);
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


  /**
   * Copies data from input stream into output stream.
   *
   * @param input the input stream to read from
   * @param output the output stream to copy to
   * @throws IOException if copying fails
   */
  public static void copy(InputStream input, OutputStream output) throws IOException {
    byte[] buffer;
    int n;

    buffer = new byte[1024];
    while ((n = input.read(buffer)) != -1)
      output.write(buffer, 0, n);
  }

  /**
   * Compresses the string.
   *
   * @param text the string to compress
   * @return the compressed string as bytes
   */
  public static byte[] compress(String text) {
    ByteArrayInputStream bis;
    ByteArrayOutputStream bos;
    GZIPOutputStream gos;

    try {
      bis = new ByteArrayInputStream(text.getBytes());
      bos = new ByteArrayOutputStream();
      gos = new GZIPOutputStream(bos);
      copy(bis, gos);
      gos.flush();
      gos.close();
      return bos.toByteArray();
    }
    catch (Exception e) {
      System.err.println("Error compressing text: " + text);
      e.printStackTrace();
      return new byte[0];
    }
  }

  /**
   * Decompresses the bytes.
   *
   * @param compressed the data to decompress
   * @return the decompressed string
   */
  public static String decompress(byte[] compressed) {
    ByteArrayInputStream bis;
    GZIPInputStream gis;
    ByteArrayOutputStream bos;

    try {
      bis = new ByteArrayInputStream(compressed);
      gis = new GZIPInputStream(bis);
      bos = new ByteArrayOutputStream();
      copy(gis, bos);
      return new String(bos.toByteArray());
    }
    catch (Exception e) {
      System.err.println("Error decompressing data:");
      e.printStackTrace();
      return "Error decompressing data: " + e;
    }
  }
}
