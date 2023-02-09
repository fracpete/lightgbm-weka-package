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
 * LightGBM.java
 * Copyright (C) 2023 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMDataset;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

/**
 * <!-- globalinfo-start -->
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -->
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start -->
 * <!-- options-end -->
 *
 * @author fracpete (fracpete at waikato dot ac dot nz)
 */
public class LightGBM
  extends AbstractClassifier
  implements TechnicalInformationHandler {

  private static final long serialVersionUID = -6138516902729782286L;

  public final static String VERSION = "3.3.2";

  public final static String PARAMETERS_URL = "https://lightgbm.readthedocs.io/en/v" + VERSION + "/Parameters.html";

  public static final int OBJECTIVE_REGRESSION = 0;
  public static final int OBJECTIVE_REGRESSION_L1 = 1;
  public static final int OBJECTIVE_HUBER = 2;
  public static final int OBJECTIVE_FAIR = 3;
  public static final int OBJECTIVE_POISSON = 4;
  public static final int OBJECTIVE_QUANTILE = 5;
  public static final int OBJECTIVE_MAPE = 6;
  public static final int OBJECTIVE_GAMMA = 7;
  public static final int OBJECTIVE_TWEEDIE = 8;
  public static final int OBJECTIVE_BINARY = 9;
  public static final int OBJECTIVE_MULTICLASS = 10;
  public static final int OBJECTIVE_MULTICLASSOVA = 11;
  public static final int OBJECTIVE_CROSS_ENTROPY = 12;
  public static final int OBJECTIVE_CROSS_ENTROPY_LAMBDA = 13;
  public static final int OBJECTIVE_LAMBDARANK = 14;
  public static final int OBJECTIVE_RANK_XENDCG = 15;

  /** the available objectives. */
  protected static final Tag[] TAGS_OBJECTIVE = {
    new Tag(OBJECTIVE_REGRESSION, "REGRESSION", "Regression"),
    new Tag(OBJECTIVE_REGRESSION_L1, "REGRESSION_L1", "Regression L1"),
    new Tag(OBJECTIVE_HUBER, "HUBER", "Huber loss"),
    new Tag(OBJECTIVE_FAIR, "FAIR", "Fair loss"),
    new Tag(OBJECTIVE_POISSON, "POISSON", "Poisson regression"),
    new Tag(OBJECTIVE_QUANTILE, "QUANTILE", "Quantile regression"),
    new Tag(OBJECTIVE_MAPE, "MAPE", "MAPE loss"),
    new Tag(OBJECTIVE_GAMMA, "GAMMA", "Gamma regression with log-link"),
    new Tag(OBJECTIVE_TWEEDIE, "TWEEDIE", "Tweedie regression with log-link"),
    new Tag(OBJECTIVE_BINARY, "BINARY", "Binary log loss classification"),
    new Tag(OBJECTIVE_MULTICLASS, "MULTICLASS", "Multi-class (softmax)"),
    new Tag(OBJECTIVE_MULTICLASSOVA, "MULTICLASSOVA", "Multi-class (one-vs-all)"),
    new Tag(OBJECTIVE_CROSS_ENTROPY, "CROSSENTROPY", "Cross-entropy"),
    new Tag(OBJECTIVE_CROSS_ENTROPY_LAMBDA, "CROSSENTROPY_LAMBDA", "Cross-entropy Lambda"),
    new Tag(OBJECTIVE_LAMBDARANK, "LAMBDA_RANK", "Lambda rank"),
    new Tag(OBJECTIVE_RANK_XENDCG, "RANK_XENDCG", "Rank Xendcg"),
  };

  /** the type of booster to use. */
  protected int m_Objective = OBJECTIVE_REGRESSION;

  /** the parameters for the booster. */
  protected String m_Parameters = "";

  /** the number of iterations to train for. */
  protected int m_NumIterations = 10;

  /** the booster instance in use. */
  protected transient LGBMBooster m_Booster = null;

  /** the built model. */
  protected String m_Model = null;

  /**
   * Returns a string describing this clusterer
   *
   * @return a description of the evaluator suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "LightGBM is a gradient boosting framework that uses tree based learning algorithms. "
      + "It is designed to be distributed and efficient.\n\n"
      + "For more information see:\n\n"
      + "https://github.com/microsoft/LightGBM\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing detailed
   * information about the technical background of this class, e.g., paper
   * reference or book this class is based on.
   *
   * @return the technical information about this class
   */
  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
    result.setValue(TechnicalInformation.Field.AUTHOR, "Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan");
    result.setValue(TechnicalInformation.Field.YEAR, "2017");
    result.setValue(TechnicalInformation.Field.TITLE, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree");
    result.setValue(TechnicalInformation.Field.BOOKTITLE, "Advances in Neural Information Processing Systems");
    result.setValue(TechnicalInformation.Field.PUBLISHER, "Curran Associates, Inc.");
    result.setValue(TechnicalInformation.Field.EDITOR, "I. Guyon and U. Von Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett");
    result.setValue(TechnicalInformation.Field.PAGES, "3149-3157");
    result.setValue(TechnicalInformation.Field.URL, "https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf");

    return result;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector<Option> 	result;
    String		desc;
    SelectedTag 	tag;
    int			i;

    result = new Vector<Option>();
    desc  = "";
    for (i = 0; i < TAGS_OBJECTIVE.length; i++) {
      tag = new SelectedTag(TAGS_OBJECTIVE[i].getID(), TAGS_OBJECTIVE);
      desc  +=   "\t" + tag.getSelectedTag().getIDStr()
	+ " = " + tag.getSelectedTag().getReadable()
	+ "\n";
    }
    result.addElement(new Option(
      "\tThe type of booster to use:\n"
	+ desc
	+ "\t(default: " + new SelectedTag(OBJECTIVE_REGRESSION, TAGS_OBJECTIVE) + ")",
      "O", 1, "-O " + Tag.toOptionList(TAGS_OBJECTIVE)));

    result.addElement(new Option(
      "\tThe parameters for the booster (blank-separated key=value pairs).\n"
	+ "\tSee: " + PARAMETERS_URL + "\n"
	+ "\t(default: none)\n",
      "P", 1, "-P <parameters>"));

    result.addElement(new Option(
      "\tThe number of iterations to train for.\n"
	+ "\t(default: none)\n",
      "I", 1, "-I <iterations>"));

    return result.elements();
  }

  /**
   * Parses the options. <p/>
   *
   <!-- options-start -->
   <!-- options-end -->
   *
   * @param options	the options to parse
   * @throws Exception 	if parsing fails
   */
  public void setOptions(String[] options) throws Exception {
    String	tmpStr;

    tmpStr = Utils.getOption('O', options);
    if (tmpStr.length() != 0)
      setObjective(new SelectedTag(tmpStr, TAGS_OBJECTIVE));
    else
      setObjective(new SelectedTag(OBJECTIVE_REGRESSION, TAGS_OBJECTIVE));

    setParameters(Utils.getOption('P', options));

    tmpStr = Utils.getOption('I', options);
    if (tmpStr.length() != 0)
      setNumIterations(Integer.parseInt(tmpStr));
    else
      setNumIterations(100);

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    List<String> result;

    result = new ArrayList<String>(Arrays.asList(super.getOptions()));

    result.add("-O");
    result.add("" + getObjective());

    if (!getParameters().trim().isEmpty()) {
      result.add("-P");
      result.add("" + getParameters());
    }

    result.add("-I");
    result.add("" + getNumIterations());

    return result.toArray(new String[0]);
  }

  /**
   * Sets the type of booster to use.
   *
   * @param value 	the type
   */
  public void setObjective(SelectedTag value) {
    if (value.getTags() == TAGS_OBJECTIVE) {
      m_Objective = value.getSelectedTag().getID();
    }
  }

  /**
   * Gets the type of booster to use.
   *
   * @return 		the type
   */
  public SelectedTag getObjective() {
    return new SelectedTag(m_Objective, TAGS_OBJECTIVE);
  }

  /**
   * Returns the tip text for this property
   *
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String objectiveTipText() {
    return "Sets the type of booster to use.";
  }

  /**
   * Sets the type of booster to use.
   *
   * @param value 	the type
   */
  public void setParameters(String value) {
    m_Parameters = value;
  }

  /**
   * Gets the parameters to use.
   *
   * @return 		the parameters
   */
  public String getParameters() {
    return m_Parameters;
  }

  /**
   * Returns the tip text for this property
   *
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String parametersTipText() {
    return "Sets the parameters to use (blank-separated key=value pairs), see: " + PARAMETERS_URL;
  }

  /**
   * Sets the number of iterations to train for.
   *
   * @param value 	the iterations
   */
  public void setNumIterations(int value) {
    if (value > 0)
      m_NumIterations = value;
  }

  /**
   * Gets the number of iterations to train for.
   *
   * @return 		the iterations
   */
  public int getNumIterations() {
    return m_NumIterations;
  }

  /**
   * Returns the tip text for this property
   *
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String numIterationsTipText() {
    return "Sets the number of iterations to train for.";
  }

  /**
   * Returns the Capabilities of this classifier.
   *
   * @return the capabilities of this object
   * @see Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities	result;

    result = new Capabilities(this);

    // attributes
    result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capabilities.Capability.DATE_ATTRIBUTES);

    // classes
    result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
    switch (m_Objective) {
      case OBJECTIVE_BINARY:
	result.enable(Capabilities.Capability.BINARY_CLASS);
	result.disable(Capabilities.Capability.UNARY_CLASS);
	break;

      case OBJECTIVE_MULTICLASS:
      case OBJECTIVE_MULTICLASSOVA:
	result.enable(Capabilities.Capability.NOMINAL_CLASS);
	result.disable(Capabilities.Capability.BINARY_CLASS);
	result.disable(Capabilities.Capability.UNARY_CLASS);
	break;

      default:
	result.enable(Capabilities.Capability.NUMERIC_CLASS);
    }

    // other
    result.setMinimumNumberInstances(1);

    return result;
  }

  /**
   * Generates a classifier. Must initialize all fields of the classifier
   * that are not being set via options (ie. multiple calls of buildClassifier
   * must always lead to the same result). Must not change the dataset
   * in any way.
   *
   * @param data set of instances serving as training data
   * @throws Exception if the classifier has not been
   *                   generated successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {
    String	parameters;
    LGBMDataset	dataset;
    int		i;

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();

    dataset = LightGBMUtils.fromInstances(data);
    parameters = "objective=" + getObjective().getSelectedTag().getIDStr()
      + " label=name:" + data.classAttribute().name();
    if (!m_Parameters.isEmpty())
      parameters += " " + m_Parameters;

    try {
      m_Booster = LGBMBooster.create(dataset, parameters);
      for (i = 0; i < m_NumIterations; i++)
	m_Booster.updateOneIter();
      m_Model = m_Booster.saveModelToString(m_NumIterations - 1, m_NumIterations - 1, LGBMBooster.FeatureImportanceType.GAIN);
    }
    catch (Exception e) {
      // m_Booster.close();  // TODO memory leak?
      dataset.close();
    }
  }

  /**
   * Initializes the booster instance.
   *
   * @throws Exception	if initialization fails
   */
  protected void initBooster() throws Exception {
    if (m_Booster == null) {
      if (m_Model != null)
	m_Booster = LGBMBooster.loadModelFromString(m_Model);
      else
	throw new IllegalStateException("No model trained?");
    }
  }

  /**
   * Classifies the given test instance. The instance has to belong to a dataset
   * when it's being classified.
   *
   * @param instance the instance to be classified
   * @return the predicted most likely class for the instance or
   *         Utils.missingValue() if no prediction is made
   * @throws Exception if an error occurred during the prediction
   */
  @Override
  public double classifyInstance(Instance instance) throws Exception {
    double[]	values;

    initBooster();

    values = LightGBMUtils.fromInstance(instance);
    return m_Booster.predictForMatSingleRow(values, PredictionType.C_API_PREDICT_NORMAL);
  }

  /**
   * Prints the all the rules of the rule learner.
   *
   * @return a textual description of the classifier
   */
  @Override
  public String toString() {
    if (m_Model == null)
      return "No model built yet.";
    else
      return m_Model;
  }

  /**
   * Returns the revision string.
   *
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }

  /**
   * Main method.
   *
   * @param args the options for the classifier
   */
  public static void main(String[] args) {
    runClassifier(new LightGBM(), args);
  }
}