
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import static java.util.Locale.filter;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.PTStemmer;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author silajs
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {

        DataSource source = new DataSource("/media/silajs/Data/GitHub/weka-classifier-ic/Arquivos samuel/min.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(0);

        DataSource source1 = new DataSource("/media/silajs/Data/GitHub/weka-classifier-ic/Arquivos samuel/min.arff");
        Instances test = source1.getDataSet();
        test.setClassIndex(1);

        //Remover estrelas (segundo atributo)
        Remove removeStarsFilter = new Remove();
        String[] removeOptions = new String[2];
        removeOptions[0] = "-R";                                    // "range"
        removeOptions[1] = "2";                                     // second attribute
        removeStarsFilter.setOptions(removeOptions);
        removeStarsFilter.setInvertSelection(false);
        removeStarsFilter.setInputFormat(data);

        Instances dataWithRemoveStar = null;
        Instances testWithRemoveStar = null;
        dataWithRemoveStar = Filter.useFilter(data, removeStarsFilter);
        testWithRemoveStar = Filter.useFilter(test, removeStarsFilter);
        dataWithRemoveStar.setClassIndex(0);
        testWithRemoveStar.setClassIndex(0);

//
        // filter -- String to word 
        StringToWordVector bagOfWordsFilter = new StringToWordVector();
        bagOfWordsFilter.setOptions(new String[]{"-C"});
        bagOfWordsFilter.setStemmer(new PTStemmer());
        bagOfWordsFilter.setAttributeIndices("last");
        bagOfWordsFilter.setInputFormat(dataWithRemoveStar);

        //Aplicando o filtro no dataset de dados
        Instances filteredData = null;
        filteredData = Filter.useFilter(dataWithRemoveStar, bagOfWordsFilter);
        filteredData.setClassIndex(0);

        //Aplicando o filtro no dataset de teste
        Instances filteredTeste = null;
        filteredTeste = Filter.useFilter(testWithRemoveStar, bagOfWordsFilter);
        filteredTeste.setClassIndex(0);
        System.out.println("Class index fileterd data " + filteredData.classIndex());

        String[] options = new String[1];
        options[0] = "-U";            // unpruned tree
        J48 tree = new J48();         // new instance of tree
        tree.setOptions(options);     // set the options
        FilteredClassifier fc = new FilteredClassifier();
        //specify filter
        fc.setFilter(bagOfWordsFilter);
        //specify base classifier
        fc.setClassifier(tree);
        dataWithRemoveStar.setClassIndex(0);
        System.out.println(dataWithRemoveStar.classAttribute());

        fc.buildClassifier(dataWithRemoveStar);   // build classifier
        System.out.println(tree.graph());
        System.out.println(tree);
        //cross validation
        Evaluation eval = new Evaluation(filteredData);
        eval.crossValidateModel(tree, filteredData, 10, new Random(1));
        //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        //System.out.println(eval.toMatrixString("\nResults\n\n"));
//        System.out.println(filteredData.toSummaryString());
//        
//        System.out.println(testWithRemoveStar.numAttributes());

        //evaluate classifier and print some statistics
//        Evaluation eval = new Evaluation(dataWithRemoveStar);
//        eval.evaluateModel(tree, filteredTeste);
//        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }

}
