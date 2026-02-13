package com.anirudhology.microgpt;

import com.anirudhology.microgpt.tokenizer.CharTokenizer;
import com.anirudhology.microgpt.tokenizer.TextCorpus;
import com.anirudhology.microgpt.training.BigramModel;

import java.util.List;

public class Runner {

    static void main() {
        TextCorpus textCorpus = new TextCorpus();
        final List<String> docs = textCorpus.readCorpus("input.txt");

        CharTokenizer tokenizer = new CharTokenizer();
        tokenizer.buildVocabulary(docs);

        BigramModel model = new BigramModel(tokenizer.getVocabularySize(), tokenizer.getBOSId(), 42L);
        model.fit(docs, tokenizer, 1.0); // alpha = 0.1 - Laplace smoothing

        double nll = model.averageNegativeLogLikelihood(docs, tokenizer);
        System.out.printf("Bigram avg NLL: %.4f%n", nll);

        System.out.println("\n--- Bigram samples ---");
        for (int i = 0; i < 20; i++) {
            System.out.printf("sample %2d: %s%n", i + 1, model.sample(tokenizer, 16));
        }
    }
}
