package com.anirudhology.microgpt.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

public class CharTokenizer {

    private final Map<Character, Integer> charToId = new HashMap<>();
    private final Map<Integer, Character> idToChar = new HashMap<>();
    private int bosId = -1;

    private boolean isVocabularyBuilt = false;

    public void buildVocabulary(List<String> docs) {
        if (isVocabularyBuilt) {
            throw new IllegalStateException("Vocabulary has already been built");
        }

        SortedSet<Character> sortedChars = new TreeSet<>();
        for (String doc : docs) {
            for (char c : doc.toCharArray()) {
                sortedChars.add(c);
            }
        }

        int id = 0;
        for (char c : sortedChars) {
            this.charToId.put(c, id);
            this.idToChar.put(id, c);
            id++;
        }

        this.bosId = id; // BOS is the last id
        this.isVocabularyBuilt = true;
    }

    public List<Integer> encode(String s) {
        // Ensure if the vocabulary is built
        ensureVocabularyIsBuilt();
        // List to store ids of characters in current string
        final List<Integer> ids = new ArrayList<>(s.length());
        // Assign each character to its specific index
        for (char c : s.toCharArray()) {
            if (!this.charToId.containsKey(c)) {
                throw new IllegalArgumentException("Unknown character: '" + c + "'");
            }
            ids.add(this.charToId.get(c));
        }
        return ids;
    }

    public String decode(List<Integer> ids) {
        // Ensure if the vocabulary is built
        ensureVocabularyIsBuilt();
        final StringBuilder decodedString = new StringBuilder(ids.size());
        for (int id : ids) {
            if (id == this.bosId) {
                continue; // Skip BOS in decoded text
            }
            if (!this.idToChar.containsKey(id)) {
                throw new IllegalArgumentException("Invalid token id: '" + id + "'");
            }
            decodedString.append(this.idToChar.get(id));
        }
        return decodedString.toString();
    }

    public int getVocabularySize() {
        ensureVocabularyIsBuilt();
        return this.charToId.size() + 1; // +1 for BOS
    }

    public int getBOSId() {
        ensureVocabularyIsBuilt();
        return this.bosId;
    }

    public List<Integer> withBOSOnBothSides(String doc) {
        ensureVocabularyIsBuilt();
        final List<Integer> out = new ArrayList<>(doc.length() + 2);
        out.add(this.bosId);
        out.addAll(encode(doc));
        out.add(this.bosId);
        return out;
    }

    private void ensureVocabularyIsBuilt() {
        if (!this.isVocabularyBuilt) {
            throw new IllegalStateException("Vocabulary is not built yet");
        }
    }
}
