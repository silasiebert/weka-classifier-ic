
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
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

        DataSource source1 = new DataSource("/media/silajs/Data/GitHub/weka-classifier-ic/Arquivos samuel/min.arff");
        Instances test = source1.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        
        
        if (test.classIndex() == -1) {
            test.setClassIndex(test.numAttributes() - 1);
        }
        //Remover estrelas (segundo atributo)
        Remove rm = new Remove();
        rm.setAttributeIndices("1"); // remove 2nd attribute
//
//        // filter -- String to word 
//        Instances filteredData = null;
//        StringToWordVector filter = new StringToWordVector();
////			filter.setOptions(new String[]{"-C"} );
//        filter.setInputFormat(data);
//        filteredData = Filter.useFilter(data, filter);
//        filteredData.setClassIndex(filteredData.numAttributes() - 1);
//        // filter -- String to word 
//        Instances filteredTeste = null;
////			filter.setOptions(new String[]{"-C"} );

//        filteredTeste = Filter.useFilter(test, filter);
//filteredTeste.setClassIndex(filteredTeste.numAttributes() - 1);
        String[] options = new String[1];
        options[0] = "-U";            // unpruned tree
        J48 tree = new J48();         // new instance of tree
        tree.setOptions(options);     // set the options
        tree.buildClassifier(data);   // build classifier
       
        
        //cross validation
       // Instances newData = ... // from somewhere
        //Evaluation eval = new Evaluation(newData);
        //J48 tree = new J48();
        //eval.crossValidateModel(tree, newData, 10, new Random(1));
        //System.out.println(eval.toSummaryString("\nResults\n\n", false));

        //evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(tree, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }

}
