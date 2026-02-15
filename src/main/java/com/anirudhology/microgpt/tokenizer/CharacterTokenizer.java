package com.anirudhology.microgpt.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * This class converts text into individual characters rather than words.
 * It builds a vocabulary from unique characters in the corpus.
 * It also performs encoding and decoding of strings and indices.
 */
public class CharacterTokenizer {

    // Stores mapping of character in the vocabulary against the index
    // of that character in the vocabulary.
    private final Map<Character, Integer> charToId = new HashMap<>();

    // Stores mapping of index against the character in the vocabulary
    // for easy references.
    private final Map<Integer, Character> idToChar = new HashMap<>();

    // BOS stands for Beginning of sentence, and it is a special
    // token marking sequence boundaries. It is assigned the last
    // index in the vocabulary.
    private int bosId = -1;

    // We need this flag to indicate if the vocabulary is built.
    // Vocabulary must be immutable so that once it is built,
    // we won't allow any other symbols in it.
    private boolean isVocabularyBuilt = false;

    /**
     * This is the beginning of the process - building the vocabulary
     *
     * @param documents - list of input text data
     */
    public void buildVocabulary(List<String> documents) {
        if (this.isVocabularyBuilt) {
            throw new IllegalStateException("Vocabulary has already been built");
        }

        // We will store the characters in the sorted order
        final SortedSet<Character> sortedCharacters = new TreeSet<>();
        for (String document : documents) {
            for (char c : document.toCharArray()) {
                sortedCharacters.add(c);
            }
        }

        // Assign ids to each of the characters
        int id = 0;
        for (char c : sortedCharacters) {
            this.charToId.put(c, id);
            this.idToChar.put(id, c);
            id++;
        }

        this.bosId = id; // BOS is the last id

        // Since the vocabulary is built, set the flag
        this.isVocabularyBuilt = true;
    }

    /**
     * Maps the input characters in the form of string into
     * their respective ids based on the vocabulary
     *
     * @param text input string
     * @return encoded ids
     */
    public List<Integer> encode(String text) {
        // Ensure if the vocabulary is built
        ensureVocabularyIsBuilt();
        // List to store ids of characters in current string
        final List<Integer> ids = new ArrayList<>(text.length());
        // Assign each character to its specific index
        for (char c : text.toCharArray()) {
            // Fail if an alien character is encountered
            if (!this.charToId.containsKey(c)) {
                throw new IllegalArgumentException("Unknown character: '" + c + "'");
            }
            ids.add(this.charToId.get(c));
        }
        return ids;
    }

    /**
     * Decodes the ids back into the string
     *
     * @param ids list of ids
     * @return decoded string
     */
    public String decode(List<Integer> ids) {
        // Ensure if the vocabulary is built
        ensureVocabularyIsBuilt();

        final StringBuilder decodedString = new StringBuilder(ids.size());
        for (int id : ids) {
            // If BOS is encountered, we skipt it
            if (id == this.bosId) {
                continue; // Skip BOS in decoded text
            }
            // Alien id is present in the list, we throw exception.
            if (!this.idToChar.containsKey(id)) {
                throw new IllegalArgumentException("Invalid token id: '" + id + "'");
            }
            decodedString.append(this.idToChar.get(id));
        }
        return decodedString.toString();
    }

    /**
     * @return size of the vocabulary
     */
    public int getVocabularySize() {
        ensureVocabularyIsBuilt();
        return this.charToId.size() + 1; // +1 for BOS
    }

    /**
     * @return id of the beginning of sentence token
     */
    public int getBOSId() {
        ensureVocabularyIsBuilt();
        return this.bosId;
    }

    /**
     * Encodes the word into ids along with BOS token at the
     * beginning and ending of sentence
     *
     * @param text to be encoded
     * @return list of ids
     */
    public List<Integer> withBOSOnBothSides(String text) {
        ensureVocabularyIsBuilt();
        final List<Integer> out = new ArrayList<>(text.length() + 2);
        out.add(this.bosId);
        out.addAll(encode(text));
        out.add(this.bosId);
        return out;
    }

    /**
     * Throws exception if someone tries to mutate the existing vocabulary
     */
    private void ensureVocabularyIsBuilt() {
        if (!this.isVocabularyBuilt) {
            throw new IllegalStateException("Vocabulary is not built yet");
        }
    }
}
