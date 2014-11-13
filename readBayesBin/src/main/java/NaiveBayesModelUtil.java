package main.java;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.io.Closeables;

public class NaiveBayesModelUtil {

	public static NaiveBayesModel materialize(String filePath) throws Exception {

		    Vector weightsPerLabel = null;
		    Vector perLabelThetaNormalizer = null;
		    Vector weightsPerFeature = null;
		    Matrix weightsPerLabelAndFeature;
		    float alphaI;

		    DataInputStream in =new DataInputStream(new FileInputStream(new File(filePath)));
		    try {
		      alphaI = in.readFloat();
		      weightsPerFeature = VectorWritable.readVector(in);
		      weightsPerLabel = new DenseVector(VectorWritable.readVector(in));
		      perLabelThetaNormalizer = new DenseVector(VectorWritable.readVector(in));

		      weightsPerLabelAndFeature = new SparseRowMatrix(weightsPerLabel.size(), weightsPerFeature.size());
		      for (int label = 0; label < weightsPerLabelAndFeature.numRows(); label++) {
		        weightsPerLabelAndFeature.assignRow(label, VectorWritable.readVector(in));
		      }
		    } finally {
		    	Closeables.close(in, true);
		    }
		    NaiveBayesModel model = new NaiveBayesModel(weightsPerLabelAndFeature, weightsPerFeature, weightsPerLabel,
		        perLabelThetaNormalizer, alphaI);
		    model.validate();
		    return model;
	}

}
