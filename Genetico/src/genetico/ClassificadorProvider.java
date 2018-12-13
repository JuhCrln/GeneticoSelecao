/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genetico;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.WiSARD;
import weka.classifiers.lazy.IBk;

/**
 *
 * @author Thiago
 */
public class ClassificadorProvider {

    private final AbstractClassifier classificador;
    
    public ClassificadorProvider(int tipo) {
        switch(tipo) {
            case 1:
                classificador = new IBk(1);
                break;
            case 2:
                classificador = new LibSVM();
                break;
            case 3:
            classificador = new WiSARD();
            break;
            default:
                classificador = new IBk(1);
        }
    }
    
    public ClassificadorProvider(String tipo) {
        switch(tipo) {
            case "knn":
                classificador = new IBk(1);
                break;
            case "svm":
                classificador = new LibSVM();
                break;
            case "mlp":
            classificador = new WiSARD();
            break;
            default:
                classificador = new IBk(1);
        }
    }
    
    public AbstractClassifier getClassificador() {
        return classificador;
    }
    
}
