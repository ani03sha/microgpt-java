package com.anirudhology.microgpt.data;

import com.anirudhology.microgpt.tokenizer.CharacterTokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Builds N-gram training dataset for language modeling.
 * <p>
 * Transforms raw text documents into supervised learning example where:
 * - Context: Fixed-size window of previous N tokens
 * - Target: The next token to predict
 */
public class NGramDatasetBuilder {

    private static final Random RANDOM = new Random(42);

    private NGramDatasetBuilder() {
    }

    /**
     * Build training examples from documents with specified context size.
     *
     * @param documents   List of text documents to process
     * @param tokenizer   Tokenizer for encoding text to IDs
     * @param contextSize Number of previous tokens to use as context (block size)
     * @return List of training examples (context, target) pairs
     */
    public static List<TrainingExample> build(List<String> documents, CharacterTokenizer tokenizer, int contextSize, boolean shuffle) {
        if (contextSize <= 0) {
            throw new IllegalArgumentException("contextSize must be positive, got: " + contextSize);
        }
        final List<TrainingExample> trainingExamples = new ArrayList<>();
        int bosId = tokenizer.getBOSId();

        for (String document : documents) {
            final List<Integer> ids = tokenizer.encode(document);
            // Targets are document characters + final BOS
            final List<Integer> targets = new ArrayList<>(ids.size() + 1);
            targets.addAll(ids);
            targets.add(bosId);

            // Initialize context with BOS padding
            int[] context = new int[contextSize];
            Arrays.fill(context, bosId);

            // Sliding window: create (context, target) pairs
            for (int target : targets) {
                trainingExamples.add(new TrainingExample(Arrays.copyOf(context, context.length), target));
                shiftLeftAppend(context, target);
            }
        }
        // Optional shuffling for better training
        if (shuffle) {
            Collections.shuffle(trainingExamples, RANDOM);
        }
        return trainingExamples;
    }

    private static void shiftLeftAppend(int[] context, int target) {
        if (context.length - 1 >= 0) {
            System.arraycopy(context, 1, context, 0, context.length - 1);
        }
        context[context.length - 1] = target;
    }
}
