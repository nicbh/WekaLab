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
import weka.classifiers.meta.Bagging;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.filters.*;
import weka.classifiers.Evaluation;
import weka.core.neighboursearch.*;
import weka.filters.unsupervised.attribute.RandomProjection;
import weka.filters.unsupervised.attribute.RandomSubset;

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
                "weka.classifiers.lazy.IBk"
//                "weka.classifiers.functions.LibLINEAR",
//                "weka.classifiers.functions.LibSVM",
//                "weka.classifiers.functions.MultilayerPerceptron"
        };

        String outputpath = "output/";
        int k = 0;
        for (int iter = 0; iter < models.length && k < 10; iter++) {
            String modelname = models[iter];
            try {
                String newstr = "7.20randomsubset";
                String name = modelname.substring(modelname.lastIndexOf('.') + 1);
                FileWriter output = new FileWriter(new File(outputpath + name + newstr + ".txt"));
//                System.out.println("-------------------------------------");
                for (int j = 0; j < files.length && k < 10; j++) {
                    String outputstr = "-------------------------------------\n";
                    String filename = files[j];
                    System.out.println("Start to run " + filename + " with " + modelname);
                    try {
                        DataSource source = new DataSource(filename);
                        Instances data = source.getDataSet();
                        if (data.classIndex() == -1)
                            data.setClassIndex(data.numAttributes() - 1);
                        int class_num = data.numClasses();

//                        Class javaclass = Class.forName(modelname);
//                        Classifier classifier = (Classifier) javaclass.newInstance();
                        IBk classifier = new IBk(7);
                        classifier.setCrossValidate(true);
                        try {
                            FilteredDistance df = new FilteredDistance();
                            Filter filter = new RandomSubset();
                            df.setFilter(filter);
                            NearestNeighbourSearch search = new LinearNNSearch();
                            search.setDistanceFunction(df);
                            classifier.setNearestNeighbourSearchAlgorithm(search);
                        } catch (Exception ex) {

                        }
                        long start = System.currentTimeMillis();
                        Evaluation eval = new Evaluation(data);
                        eval.crossValidateModel(classifier, data, 10, new Random(1));
                        long elapsed = System.currentTimeMillis() - start;

//                        System.out.println("10 folds CV on " + filename + " with model " + modelname);
//                        System.out.println("Elapsed " + elapsed + "ms. ");
//                        System.out.println(eval.toSummaryString());
                        outputstr += "10 folds CV on " + filename + " with model " + modelname + "\n";
                        outputstr += "Elapsed " + elapsed + "ms.\n";
                        outputstr += eval.toSummaryString() + "\n";
                        for (int i = 0; i < class_num; i++) {
//                            System.out.println("ROC area for class " + data.classAttribute().value(i) + " is: " + eval.areaUnderROC(i));
                            outputstr += "ROC area for class " + data.classAttribute().value(i) + " is: " + eval.areaUnderROC(i) + "\n";
                        }
//                        System.out.println();
                        outputstr += "\n";

                        System.out.println("bagging");
                        Bagging bagger = new Bagging();
                        bagger.setClassifier(classifier);
                        bagger.setBagSizePercent(20);

                        start = System.currentTimeMillis();
                        eval = new Evaluation(data);
                        eval.crossValidateModel(bagger, data, 10, new Random(1));
                        elapsed = System.currentTimeMillis() - start;

//                        System.out.println("10 folds CV on " + filename + " with model " + modelname);
//                        System.out.println("Elapsed " + elapsed + "ms. ");
//                        System.out.println(eval.toSummaryString());
                        outputstr += "10 folds CV on " + filename + " with model " + modelname + " with bagging.\n";
                        outputstr += "Elapsed " + elapsed + "ms.\n";
                        outputstr += eval.toSummaryString() + "\n";
                        for (int i = 0; i < class_num; i++) {
//                            System.out.println("ROC area for class " + data.classAttribute().value(i) + " is: " + eval.areaUnderROC(i));
                            outputstr += "ROC area for class " + data.classAttribute().value(i) + " is: " + eval.areaUnderROC(i) + "\n";
                        }
                        System.out.println();
                        outputstr += "\n";
                    } catch (Exception ex) {
                        outputstr += ex.toString();
                        ex.printStackTrace();
                        k++;
                    }
                    outputstr = outputstr.replaceAll("\\n", System.getProperty("line.separator"));
                    output.write(outputstr);
                }
                k = 0;
                output.close();
            } catch (IOException ex) {
                ex.printStackTrace();
                iter--;
                k++;
            }
        }
    }

    static class myIBk extends IBk implements Randomizable {
        int seed;
        Random m_random;

        myIBk(int n) {
            super(n);
            this.setCrossValidate(true);
        }

        @Override
        public int getSeed() {
            return seed;
        }

        @Override
        public void setSeed(int i) {
            m_random = new Random(i);
            double random = m_random.nextDouble();
            try {
                FilteredDistance df = new FilteredDistance();
//                if (random < 0.1)
//                    df.setDistance(new ManhattanDistance());
//                else if (random < 0.9)
//                    df.setDistance(new EuclideanDistance());
//                else if (random < 0.95)
//                    df.setDistance(new ChebyshevDistance());
//                else {
//                    MinkowskiDistance md = new MinkowskiDistance();
//                    md.setOrder(1 / (m_random.nextDouble() * 100 + 0.000001));
//                    df.setDistance(md);
//                }
                Filter filter = new RandomSubset();
                df.setFilter(filter);
                NearestNeighbourSearch search = new LinearNNSearch();
                this.setNearestNeighbourSearchAlgorithm(search);
                search.setDistanceFunction(df);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            seed = i;
        }
    }
}
