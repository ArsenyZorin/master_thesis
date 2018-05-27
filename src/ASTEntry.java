package trees;

import com.intellij.lang.ASTNode;
import com.intellij.openapi.editor.Document;
import com.intellij.psi.impl.source.tree.CompositeElement;
import com.intellij.psi.impl.source.tree.LeafElement;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class ASTEntry {
    public String nodeName;
    public final short nodeIndex;
    public int sourceStart;
    public int sourceEnd;
    public final ASTEntry parent;
    public List<ASTEntry> children;
    public final boolean isTerminal;
    public String text;
    public String filePath;

    /**
     * Sets file path to tree
     * @param filePath File path
     */
    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

	/**
	 * Removes useless nodes
	 * @param blackList List of nodes to remove
	 */
    public ASTEntry removeSpaces(List<String> blackList){
        children = children.stream().filter(p->!blackList.contains(p.nodeName)).collect(Collectors.toList());
        for(ASTEntry child : children)
            child.removeSpaces(blackList);
        return this;
    }

    /**
     * Counts nodes amount
     * @return Amount of nodes
     */
    public int getNodesAmount(){
        int amount = 0;
        if(children.size() == 0) {
            amount++;
            return amount;
        }
        for(ASTEntry node : children){
            amount += node.getNodesAmount();
        }
        return amount;
    }

    /**
     * Gets String list of all tokens
     * @return list of all tokens
     */
    public List<String> getAllTokensList(){
        List<String> nodesTokens = new ArrayList<>();
        if(children.size() == 0){
            nodesTokens.add(nodeName);
            return nodesTokens;
        }
        for(ASTEntry node : children){
            nodesTokens.addAll(node.getAllTokensList());
        }
        return nodesTokens;
    }

    /**
     * Mutates method
     * @param blackList All tokens that could be mutated
     */
    public void mutate(List<String> blackList){
        Random rnd = new Random();
        int func = rnd.nextInt(3);

        if(children.size() < 5)
            func = 2;

        switch(func) {
            case 0:
                deleteNode(blackList);
                break;
            case 1:
                copyNode();
                break;
            default:
                break;
        }
    }

    /**
     * Mutates method by deleting lines
     * @param blackList All tokens that could be mutated
     */
    private void deleteNode(List<String> blackList){
        int[] pos = getStartEndMethod();
        Random rnd = new Random();

        int amountOfLines = 0;
        if(children.size() < 10)
            amountOfLines = rnd.nextInt(children.size() - 3) + 1;
        else
            amountOfLines = rnd.nextInt(10) + 1;

        for (int i = 0; i < amountOfLines; i++) {
            int line;
            if(pos[1] == 0)
                line = rnd.nextInt(pos[1]) + pos[0] + 1;
            else
                line = rnd.nextInt(pos[1] - 1) + pos[0] + 1;

            children.forEach(child -> {
                if (children.indexOf(child) == line) {
                    child.nodeName = blackList.stream()
                            .filter(p -> p.contains("WHITE")).findFirst().get();
                }
            });
        }
    }

    /**
     * Mutates method by copying and pasting lines
     */
    private void copyNode(){
        int[] pos = getStartEndMethod();
        Random rnd = new Random();
        int amountOfLines = rnd.nextInt(children.size() - 2);

        for(int i = 0; i < amountOfLines; i++){
            int copyLine;

            if (pos[1] == 0)
                copyLine = rnd.nextInt(pos[1]) + pos[0] + 1;
            else
                copyLine = rnd.nextInt(pos[1] - 1) + pos[0] + 1;
            int pasteLine = rnd.nextInt(pos[1] - 1) + pos[0] + 1;

            while (pasteLine == copyLine){
                pasteLine = rnd.nextInt(pos[1] - 1) + pos[0] + 1;
            }

            ASTEntry copyNode = null;
            for(ASTEntry child : children){
                if (children.indexOf(child) == copyLine) {
                    copyNode = new ASTEntry(child);
                    break;
                }
            }
            children.add(pasteLine, copyNode);
        }
    }

    /**
     * Gets start and end lines of method
     * @return array with two numbers [start, end]
     */
    private int[] getStartEndMethod(){
        int pos[] = new int[2];
        children.forEach(p->{
            if("LBRACE".equals(p.nodeName))
                pos[0] = children.indexOf(p);
            if("RBRACE".equals(p.nodeName))
                pos[1] = children.indexOf(p);
        });

        if(pos[1] == 0)
            System.out.println("Something went wrong");
        return pos;
    }

    @Override
    public String toString() {
        return shiftedString("");
    }

}
