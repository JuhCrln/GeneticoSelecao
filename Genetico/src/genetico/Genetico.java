package genetico;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import org.apache.commons.lang3.ArrayUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MLPClassifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.WiSARD;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.converters.LibSVMLoader;
import weka.core.converters.LibSVMSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Genetico {
    
    static private Instances train;
    static private Instances test;
    
    static private void carregaDados() throws Exception {
        DataSource source = new DataSource("C:\\Users\\Julliana\\Downloads\\bases_arff\\hand-based-signature.arff");
        Instances D = source.getDataSet();
        if (D.classIndex() == -1) {
            D.setClassIndex(D.numAttributes() - 1);
        }
        
        train = new Instances(D, 0);
        test = new Instances(D, 0);
        
        int cont = 0;
        for (int i = 0; i < D.numInstances(); i++) {
            cont++;
            if(cont == 3) {
                cont = 0;
                test.add(D.get(i));
            } else {
                train.add(D.get(i));
            }
        }
        
        if (train.classIndex() == -1) {
            train.setClassIndex(train.numAttributes() - 1);
        }
        
        if (test.classIndex() == -1) {
            test.setClassIndex(test.numAttributes() - 1);
        }
    }
    
    static private int[] randn(int n) {
        int numeros[] = new int[n];
        for (int i = 0; i < n; i++) {
            numeros[i] = ThreadLocalRandom.current().nextInt(0, 2);
        }
        return numeros;
    }

    static private int randi(int a, int b) {
        return ThreadLocalRandom.current().nextInt(a, b + 1);
    }

    static private double randd(int a, int b) {
        return ThreadLocalRandom.current().nextDouble(a, b);
    }

    static private class Cromossomo {

        public int x[];
        public double valor;

        public Cromossomo(int n) {
            x = new int[n];
            valor = 100.0;
        }

        public Cromossomo(int x[], double valor) {
            this.x = x;
            this.valor = valor;
        }
    }

    static private double fitness(Cromossomo c) {
        int n_atrib = 1;
        for (int i = 0; i < c.x.length; i++) {
            if(c.x[i] == 1) {
                n_atrib++;
            }
        }
        int indices[] = new int[n_atrib];
        int cont = 0;
        for (int i = 0; i < c.x.length; i++) {
            if(c.x[i] == 1) {
                indices[cont] = i;
                cont++;
            }
        }
        indices[indices.length-1] = train.numAttributes()-1;
        Instances mytrain = new Instances(train);
        Instances mytest = new Instances(test);
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(indices);
        removeFilter.setInvertSelection(true);
        try {
            removeFilter.setInputFormat(mytrain);
            mytrain = Filter.useFilter(mytrain, removeFilter);
            removeFilter.setInputFormat(mytest);
            mytest = Filter.useFilter(mytest, removeFilter);
            if (mytrain.classIndex() == -1) {
                mytrain.setClassIndex(mytrain.numAttributes() - 1);
            }

            if (mytest.classIndex() == -1) {
                mytest.setClassIndex(mytest.numAttributes() - 1);
            }
           
            ClassificadorProvider classprov = new ClassificadorProvider("svm");
            AbstractClassifier classifier = classprov.getClassificador();
            classifier.buildClassifier(mytrain);
            Evaluation ev = new Evaluation(mytest);
            ev.evaluateModel(classifier, mytest);
            return ev.pctIncorrect();
        } catch(Exception ex) {
            System.out.println(ex);
        }
        return 100.0;
    }

    static private Cromossomo mutacao(Cromossomo c) {
        int n = c.x.length - 1;
        Cromossomo c_mut = new Cromossomo(c.x.clone(), c.valor);
        int pos1 = ThreadLocalRandom.current().nextInt(0, n + 1);
        int pos2 = ThreadLocalRandom.current().nextInt(0, n + 1);
        c_mut.x[pos1] = Math.abs(c_mut.x[pos1] - 1);
        c_mut.x[pos2] = Math.abs(c_mut.x[pos2] - 1);
        c_mut.valor = fitness(c_mut);
        return c_mut;
    }

    static private Cromossomo[] cruzamento(Cromossomo c1, Cromossomo c2) {
        int n = c1.x.length;
        int pos_corte = ThreadLocalRandom.current().nextInt(1, n - 1);
        Cromossomo filho1 = new Cromossomo(n);
        Cromossomo filho2 = new Cromossomo(n);
        for (int i = 0; i < n; i++) {
            if (i <= pos_corte) {
                filho1.x[i] = c1.x[i];
            } else {
                filho1.x[i] = c2.x[i];
            }
        }
        for (int i = 0; i < n; i++) {
            if (i <= pos_corte) {
                filho2.x[i] = c2.x[i];
            } else {
                filho2.x[i] = c1.x[i];
            }
        }
        filho1.valor = fitness(filho1);
        filho2.valor = fitness(filho2);
        Cromossomo resultado[] = {filho1, filho2};
        return resultado;
    }

    static private Cromossomo[] criapopinicial(int tamanho, int N) {
        Cromossomo pop[] = new Cromossomo[N];
        for (int i = 0; i < N; i++) {
            pop[i] = new Cromossomo(randn(tamanho), 0.0);
            pop[i].valor = fitness(pop[i]);
        }
        return pop;
    }

    static private Cromossomo busca(Cromossomo c) {
        int n = c.x.length - 1;
        Cromossomo melhor = c;
        for (int i = 0; i < n; i++) {
            Cromossomo c_tmp = new Cromossomo(c.x.clone(), c.valor);
            c_tmp.x[i] = Math.abs(c_tmp.x[i] - 1);
            c_tmp.valor = fitness(c_tmp);
            if (c_tmp.valor < melhor.valor) {
                melhor = c_tmp;
            }
        }
        return melhor;
    }

    static private double[] calc_prob(Cromossomo pop[]) {
        int n = pop.length;
        double maior = 0.0;
        double fit_vetor[] = new double[n];
        for (int i = 0; i < n; i++) {
            fit_vetor[i] = pop[i].valor;
            if (fit_vetor[i] > maior) {
                maior = fit_vetor[i];
            }
        }
        maior = 1.05 * maior;
        double soma = 0.0;
        for (int i = 0; i < n; i++) {
            fit_vetor[i] = maior - fit_vetor[i];
            soma += fit_vetor[i];
        }
        for (int i = 0; i < n; i++) {
            fit_vetor[i] = fit_vetor[i] / soma;
        }
        return fit_vetor;
    }

    static private int rouletteSelect(double prob[]) {
        double weight_sum = 0.0;
        for (int i = 0; i < prob.length; i++) {
            weight_sum += prob[i];
        }
        double value = ThreadLocalRandom.current().nextDouble(1) * weight_sum;
        for (int i = 0; i < prob.length; i++) {
            value -= prob[i];
            if (value < 0) {
                return i;
            }
        }
        return prob.length - 1;
    }

    static private Cromossomo[] selecao(Cromossomo pop[], int tam_pop) {
        Cromossomo pop_nova[] = new Cromossomo[tam_pop];
        for (int i = 0; i < tam_pop; i++) {
            double prob[] = calc_prob(pop);
            int pos = rouletteSelect(prob);
            pop_nova[i] = pop[pos];
            pop = ArrayUtils.removeElement(pop, pos);
        }
        return pop_nova;
    }
    
    static private void executa() throws Exception {
        System.out.println("");
        carregaDados();    
        int natrib = train.numAttributes()-1;
        int poptam = 30;
        Cromossomo pop[] = criapopinicial(natrib,poptam);
        System.out.println("População inicial criada.");
        Cromossomo melhor = new Cromossomo(natrib);
//        int c[] = new int[natrib];
//        for (int i = 0; i < natrib; i++) {
//            c[i] = 1;
//        }
//        int c[] = {0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,0, 0, 0};
//        melhor.x = c;
//        melhor.valor = fitness(melhor);
        for (int gerac = 0; gerac < 15; gerac++) {
            Cromossomo pop_nova[] = new Cromossomo[3*poptam];
            int pop_nova_i = 0;
            for (int i = 0; i < poptam; i++) {
                int escolha = randi(0,poptam-1);
                Cromossomo c1 = pop[i];
                Cromossomo c2 = pop[escolha];
                Cromossomo filhos[] = cruzamento(c1,c2);
                /*Busca*/
                //Cromossomo f1 = filhos[0];
                //Cromossomo f2 = filhos[1];
                Cromossomo f1 = busca(filhos[0]);
                Cromossomo f2 = busca(filhos[1]);
                if(randd(0,1) < 0.03) {
                    Cromossomo f1tmp = mutacao(f1);
                    Cromossomo f2tmp = mutacao(f2);
                    if (f1tmp.valor < f1.valor) {
                        f1 = f1tmp;
                    }
                    if (f2tmp.valor < f2.valor) {
                        f2 = f2tmp;
                    }
                }
                pop_nova[pop_nova_i] = f1;
                pop_nova_i++;
                pop_nova[pop_nova_i] = f2;
                pop_nova_i++;
            }
            for (int i = pop_nova_i; i < pop_nova.length; i++) {
                pop_nova[i] = pop[i-pop_nova_i];
            }
            for (Cromossomo pop_nova1 : pop_nova) {
                if (melhor.valor > pop_nova1.valor) {
                    melhor = pop_nova1;
                }
            }
            pop = selecao(pop_nova,poptam);
            pop[poptam-1] = melhor;
            System.out.println("Geração " + (gerac+1) + ", valor objetivo = " + melhor.valor);
        }
        
        System.out.println(Arrays.toString(melhor.x));
        System.out.print("[");
        for (int i = 0; i < natrib-1; i++) {
            if(melhor.x[i] == 1) {
                System.out.print(" "+train.attribute(i).name());
            }
        }
        System.out.println(" ]");
        System.out.println("Acurácia: "+(100.0-melhor.valor)+"%");
    }

    public static void main(String[] args) throws Exception {
        //for (int i = 0; i < 10; i++) {
            executa();
        //}
    }
}
