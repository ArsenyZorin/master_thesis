package trees;

import com.intellij.codeInsight.ContainerProvider;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PsiGen {
    private final PsiFileFactory fileFactory;

    public PsiGen() {
        Extensions.getRootArea().registerExtensionPoint(
                ContainerProvider.EP_NAME.getName(),
                JavaContainerProvider.class.getCanonicalName()
        );

        final CoreApplicationEnvironment appEnv = new JavaCoreApplicationEnvironment(() -> {
        });

        final CoreProjectEnvironment prjEnv = new JavaCoreProjectEnvironment(() -> {
        }, appEnv) {
            @Override
            protected void preregisterServices() {
                Extensions.getArea(myProject).registerExtensionPoint(
                        PsiTreeChangePreprocessor.EP_NAME.getName(),
                        PsiModificationTrackerImpl.class.getCanonicalName()
                );
                Extensions.getArea(myProject).registerExtensionPoint(
                        PsiElementFinder.EP_NAME.getName(),
                        PsiElementFinderImpl.class.getCanonicalName()
                );
                myProject.registerService(
                        PsiNameHelper.class,
                        PsiNameHelperImpl.class
                );
            }
        };

        this.fileFactory = PsiFileFactory.getInstance(prjEnv.getProject());

    }

    private ASTEntry convertSubtree(ASTNode node, ASTEntry parent, Document doc, String fileName) {
        ASTEntry rootEntry = new ASTEntry(node, parent, doc);
        rootEntry.setFilePath(fileName);
        for (ASTNode child : node.getChildren(null)) {
            ASTEntry entry = convertSubtree(child, rootEntry, doc, fileName);
            rootEntry.children.add(entry);
        }
        return rootEntry;
    }

    private ASTEntry convert(PsiFile file, String fileName) {
        Document doc = file.getViewProvider().getDocument();
        FileASTNode startNode = file.getNode();
        return convertSubtree(startNode, null, doc, fileName);
    }

    private List<String> elemArrayToString(IElementType[] et) {
        return Stream.of(et).map(IElementType::toString).collect(Collectors.toList());
    }

    private void fillFile(ASTEntry root, Map<Integer, String> file) {
        file.merge(root.sourceStart, root.nodeName, (v, s) -> v.concat(" " + s));
        for (ASTEntry child : root.children) {
            fillFile(child, file);
        }
    }

    private String getASTText(ASTEntry root) {
        Map<Integer, String> file = new HashMap<>();
        for (ASTEntry child : root.children) {
            fillFile(child, file);
        }
        return file.entrySet().stream().map(Map.Entry::getValue).collect(Collectors.joining(" "));
    }

    private Object getField(Field f) {
        try {
            return f.get(null);
        } catch (NullPointerException | IllegalAccessException ex) {
            ex.printStackTrace();
        }
        return null;
    }

    private PsiFile parse(final String sourceCode) {
        return fileFactory.createFileFromText(JavaLanguage.INSTANCE, sourceCode);
    }

    public List<String> getAllAvailableTokens() {
        Set<String> tokens = new HashSet<>();
        for (Field f : ElementType.class.getDeclaredFields())
            tokens.addAll(elemArrayToString(((TokenSet) getField(f)).getTypes()));
        for (Class cl : ElementType.class.getInterfaces()) {
            for (Field f : cl.getDeclaredFields()) {
                Object value = getField(f);
                if (value instanceof TokenSet)
                    tokens.addAll(elemArrayToString(((TokenSet) value).getTypes()));
                else
                    tokens.add(value.toString());
            }
        }
        tokens.add("java.FILE");
        return new ArrayList<>(tokens);
    }

    public String parsePSIText(String filename) {
        ASTEntry root = parseFile(filename);
        return getASTText(root);
    }

    public ASTEntry parseFile(String filename) {
        try {
            if(filename.contains("\t")) {
                String[] path = filename.split("\t");
                return convert(parse(
                        String.join("\n", FileUtils.readFileToString(new File(path[0]), "UTF-8"))
                ), path[1]);
            } else {
                return convert(parse(
                        String.join("\n", FileUtils.readFileToString(new File(filename), "UTF-8"))
                ), filename);
            }
        } catch (IOException e) {
            System.out.println();
            e.printStackTrace();
            return null;
        } catch (AssertionError ex){
            return null;
        }
    }
}
