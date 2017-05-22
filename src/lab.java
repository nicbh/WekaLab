/**
 * Created by nicbh on 2017/5/22.
 */

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.WekaPackageManager;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

public class lab {
    public static void main(String[] args) {
        String filepath = "smalldata";
        String[] files = null;

        File file = new File(filepath);
        if (file.isDirectory()) {
            String[] filelist = file.list();
            ArrayList<String> temp = new ArrayList<String>();
            for (String filename : filelist) {
                if (filename.endsWith(".arff")) {
                    File readfile = new File(filepath + "/" + filename);
                    temp.add(readfile.getPath());
                }
            }
            files = new String[temp.size()];
            temp.toArray(files);
        }
        for (String name : files)
            System.out.println(name);
        System.out.println();

        String[] models = {
//                "weka.classifiers.trees.J48",
//                "weka.classifiers.bayes.NaiveBayes",
//                "weka.classifiers.functions.LibSVM",
//                "weka.classifiers.functions.LibLINEAR",
//                "weka.classifiers.functions.MultilayerPerceptron",

                "weka.classifiers.functions.Dl4jMlpClassifier"
//                "weka.classifiers.lazy.IBk"
        };
        for (String modelname : models) {
            System.out.println("-------------------------------------");
            System.out.println("Start to run " + modelname);
            for (String filename : files) {
                try {
                    DataSource source = new DataSource(filename);
                    Instances data = source.getDataSet();
                    if (data.classIndex() == -1)
                        data.setClassIndex(data.numAttributes() - 1);
                    int class_num = data.numClasses();

                    //WekaPackageManager.loadPackages( false, true, false );
                    Class javaclass = Class.forName(modelname);
                    Classifier classifier = (Classifier) javaclass.newInstance();

                    long start = System.currentTimeMillis();
                    Evaluation eval = new Evaluation(data);
                    eval.crossValidateModel(classifier, data, 10, new Random(1));
                    long elapsed = System.currentTimeMillis() - start;

                    System.out.println("10 folds CV on " + filename + " with model " + modelname);
                    System.out.println("Elapsed " + elapsed + "ms. ");
                    System.out.println(eval.toSummaryString());
                    for (int i = 0; i < class_num; i++) {
                        System.out.println("ROC area for class " + data.classAttribute().value(i) + " is: " + eval.areaUnderROC(i));
                    }
                    System.out.println();

                } catch (Exception ex) {

                }
            }
        }
    }
}
