
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;

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
        
        
        
        DataSource source = new DataSource("/home/silajs/Downloads/Atividade P2/dataset/corpus_google_play/all_stage1.csv");
        Instances data = source.getDataSet();

        DataSource source1 = new DataSource("/home/silajs/Downloads/Atividade P2/dataset/corpus_10k/corpus_A.csv");
        Instances test = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        String[] options = new String[1];
        options[0] = "-U";            // unpruned tree
        J48 tree = new J48();         // new instance of tree
        tree.setOptions(options);     // set the options
        tree.buildClassifier(data);   // build classifier

        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(tree, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }

}
