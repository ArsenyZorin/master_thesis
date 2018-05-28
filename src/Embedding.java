package preproc;

import methods.Pairs;
import arguments.EvalType;
import com.google.gson.Gson;
import gitrepos.Repository;
import org.deeplearning4j.models.word2vec.Word2Vec;
import postgresql.PostgreSQL;
import trees.ASTEntry;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class Embedding {
    Word2Vec mainVec;
    TreeMutator treeMutator;
    String workingDir;
    File train_repo;

    public Embedding(TreeMutator treeMutator, String evalType, String outputDir) {
        this.treeMutator = treeMutator;
        this.workingDir = outputDir + "/networks/word2vec";

        if(EvalType.FULL.toString().toUpperCase().equals(evalType.toUpperCase()))
            train();
        else {
            File mainEmb = new File(workingDir + "/word2Vec");
            if (mainEmb.exists()) {
                System.out.println("Tokens file was found. Reading values from it");
                mainVec = WordVectorSerializer.readWord2VecModel(mainEmb);
            } else {
                train();
            }
        }
    }

    /***
     * Method for word2vec model training.
     * Uses openjdk from siemens.spbpu.com for training model
     */
    public void train() {
        Repository repository = new Repository("/tmp/w2v_train",
                 "https://siemens.spbpu.com/arseny/openjdk.git");

        train_repo = repository.getRepoFile();
        System.out.println("Additional analysis : " + train_repo.getAbsolutePath());
        List<ASTEntry> tree = treeMutator.analyzeDir(train_repo.getAbsolutePath(), null);
        AbstractMap<String, String> treeTokens = new HashMap<>();

        for (ASTEntry token : tree) {
            String ident = String.format("Path: %s Start: %d End: %d", token.filePath, token.sourceStart, token.sourceEnd);
            treeTokens.put(ident, token.getAllTokensString());
            System.out.print("\rTokensString: " + tree.indexOf(token) + "/" + tree.size());
        }

        TokenizerFactory t = new DefaultTokenizerFactory();
        SentenceIterator iter = new CollectionSentenceIterator(treeTokens.values());
        System.out.println("\nBuilding model...");
        mainVec = new Word2Vec.Builder()
                .minWordFrequency(1).iterations(1)
                .layerSize(100).seed(42).windowSize(5)
                .iterate(iter).tokenizerFactory(t).build();
        System.out.println("Fitting Word2Vec model...");
        mainVec.fit();
        try {
            WordVectorSerializer.writeWord2VecModel(mainVec, workingDir + "/word2Vec");
            gsonSerialization(mainVec.getLookupTable().getWeights(), workingDir + "/tokensWeight");
            ArrayList<double[]> weights = new ArrayList<>();
            for (int j = 0; j < mainVec.getVocab().numWords(); j++)
                weights.add(mainVec.getWordVector(mainVec.getVocab().wordAtIndex(j)));

            gsonSerialization(weights, workingDir + "/pretrainedWeights");
        } catch (Exception ex) {
            System.out.println(ex.toString());
        }
        System.out.println("ADDITIONAL ANALYSIS COMPLETE");
    }

    /**
     * Creates vector representations of tokens based on word2veec model. Uses BigCloneBench
     * @param files Map of files. Where key - path to file and start-end lines of method and value - list of tokens
     * @param ds DataSet with pairs from database (Clones and non-clones)
     * @param file_name Name of file where to save representations
     */
    private void createFromBcb(Map<String, List<ASTEntry>> files, List<Pairs> ds, String file_name){
        StringBuilder sb = new StringBuilder();
        for(Pairs pair : ds){
            List<ASTEntry> first_list = files.get(pair.getFirst().getPath());
            List<ASTEntry> second_list = files.get(pair.getSecond().getPath());

            if (first_list == null || second_list == null)
                continue;
            ASTEntry first = null;
            ASTEntry second = null;

            if(pair.getSecond().getPath().equals(pair.getFirst().getPath())) {
                for(ASTEntry entry : first_list){
                    if(entry.sourceStart >= pair.getFirst().getStart() &&
						entry.sourceEnd <= pair.getFirst().getEnd() ||
                         entry.sourceStart <= pair.getFirst().getStart() &&
						 entry.sourceEnd >= pair.getFirst().getEnd())
                        first = entry;
                    if(entry.sourceStart >= pair.getSecond().getStart() &&
						entry.sourceEnd <= pair.getSecond().getEnd() ||
                        entry.sourceStart <= pair.getSecond().getStart() && 
						entry.sourceEnd >= pair.getSecond().getEnd())
                        second = entry;
                }
            } else {
                for(ASTEntry entry: first_list)
                    if(Math.abs(pair.getFirst().getStart() - entry.sourceStart) <= 2 &&
                            Math.abs(pair.getFirst().getEnd() - entry.sourceEnd) <= 2) {
                        first = entry;
                        break;
                    }
                for(ASTEntry entry: second_list)
                    if(Math.abs(pair.getSecond().getStart() - entry.sourceStart) <= 2 &&
                            Math.abs(pair.getSecond().getEnd() - entry.sourceEnd) <= 2) {
                        second = entry;
                        break;
                    }
            }
            if (first == null || second == null)
                continue;

            List<Integer> first_tokens = new ArrayList<>();
            for (String token : first.getAllTokensList())
                first_tokens.add(mainVec.indexOf(token));

            List<Integer> second_tokens = new ArrayList<>();
            for (String token : second.getAllTokensList())
                second_tokens.add(mainVec.indexOf(token));

            String first_inds = first_tokens.stream().map(Object::toString)
                    .collect(Collectors.joining(", "));
            String sec_inds = second_tokens.stream().map(Object::toString)
                    .collect(Collectors.joining(", "));

            sb.append(String.format("%s=[%s]\t%s=[%s]\t%d\n", pair.getFirst().toString(),
                    first_inds, pair.getSecond().toString(), sec_inds, pair.isClones() ? 0:1));
        }
        writeObject(sb.toString(), file_name);
    }

    /**
     * Writes formatted string to specified path
     * @param object Formatted string
     * @param save_path Path where to save object
     */
    private void writeObject(String object, String save_path){
        FileWriter fw = null;
        BufferedWriter bw = null;
        try {
            fw = new FileWriter(save_path);
            bw = new BufferedWriter(fw);
            bw.write(object);
        } catch (IOException ex){
            ex.printStackTrace();
        } finally {
            exceptionHandling(fw, bw);
        }
    }

    /**
     * Finally block for writing to file
     * @param fw FileWriter that has to be closed
     * @param bw BufferedWriter that has to be closed
     */
    private void exceptionHandling(FileWriter fw, BufferedWriter bw) {
        try {
            if (bw != null)
                bw.close();
            if (fw != null)
                fw.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
